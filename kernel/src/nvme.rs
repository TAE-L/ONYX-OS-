//! NVMe driver (NVM command set, namespace 1, 4 KiB memory pages).
//!
//! One I/O queue pair per CPU (as many as the controller grants), so CPUs never
//! contend on a submission queue. Completions are polled with interrupts off for
//! the duration of the command: simple and fast at NVMe latencies, but it holds
//! the CPU until the device answers. Interrupt-driven completion (MSI-X) is the
//! planned next step.

#![allow(dead_code)] // the VFS (Phase 4c) is the main consumer

use crate::arch::time;
use crate::block::{self, BlockDevice, BlockError};
use crate::mm::{self, PAGE_SIZE, frame, paging};
use crate::pci::{self, Device};
use crate::percpu::cpu_index;
use crate::sched;
use crate::sync::IrqMutex;
use alloc::{format, string::String, sync::Arc, vec::Vec};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{Ordering::SeqCst, fence};

const REG_CAP: u64 = 0x00;
const REG_CC: u64 = 0x14;
const REG_CSTS: u64 = 0x1C;
const REG_AQA: u64 = 0x24;
const REG_ASQ: u64 = 0x28;
const REG_ACQ: u64 = 0x30;
const DOORBELLS: u64 = 0x1000;

const ADMIN_DEPTH: u16 = 16;
const IO_DEPTH: u16 = 64;
/// Largest single transfer: one PRP list page covers 512 pages after the first.
const MAX_TRANSFER: usize = 256 * 1024;
const TIMEOUT_NS: u64 = 5_000_000_000;

#[derive(Debug)]
pub enum InitError {
    NoBar,
    Mmio,
    Unsupported(&'static str),
    Timeout(&'static str),
    Command(&'static str, u16),
    NoMemory,
}

struct Queue {
    id: u16,
    depth: u16,
    sq: *mut u8,
    cq: *mut u8,
    sq_phys: u64,
    cq_phys: u64,
    sq_tail: u16,
    cq_head: u16,
    phase: bool,
    next_cid: u16,
    sq_db: *mut u32,
    cq_db: *mut u32,
    /// One page for PRP lists of this queue's (single in-flight) command.
    prp_list: *mut u64,
    prp_list_phys: u64,
}

// SAFETY: the raw pointers address device and DMA memory owned by the queue and
// are only used while holding its lock.
unsafe impl Send for Queue {}

#[derive(Clone, Copy, Default)]
struct Command {
    opcode: u8,
    nsid: u32,
    prp1: u64,
    prp2: u64,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
}

struct Completion {
    result: u32,
    status: u16,
}

impl Queue {
    fn new(regs: u64, dstrd: u64, id: u16, depth: u16, with_prp: bool) -> Result<Queue, InitError> {
        let sq_phys = frame::alloc_zeroed().ok_or(InitError::NoMemory)?;
        let cq_phys = frame::alloc_zeroed().ok_or(InitError::NoMemory)?;
        let (prp_list_phys, prp_list) = if with_prp {
            let p = frame::alloc_zeroed().ok_or(InitError::NoMemory)?;
            (p, mm::phys_to_virt(p).cast::<u64>())
        } else {
            (0, core::ptr::null_mut())
        };
        let stride = 4u64 << dstrd;
        Ok(Queue {
            id,
            depth,
            sq: mm::phys_to_virt(sq_phys),
            cq: mm::phys_to_virt(cq_phys),
            sq_phys,
            cq_phys,
            sq_tail: 0,
            cq_head: 0,
            phase: true,
            next_cid: 1,
            sq_db: (regs + DOORBELLS + u64::from(id) * 2 * stride) as *mut u32,
            cq_db: (regs + DOORBELLS + (u64::from(id) * 2 + 1) * stride) as *mut u32,
            prp_list,
            prp_list_phys,
        })
    }

    /// Submit one command and poll for its completion.
    fn execute(&mut self, c: &Command) -> Result<Completion, &'static str> {
        let cid = self.next_cid;
        self.next_cid = self.next_cid.wrapping_add(1).max(1);
        let mut raw = [0u32; 16];
        raw[0] = u32::from(c.opcode) | u32::from(cid) << 16;
        raw[1] = c.nsid;
        raw[6] = c.prp1 as u32;
        raw[7] = (c.prp1 >> 32) as u32;
        raw[8] = c.prp2 as u32;
        raw[9] = (c.prp2 >> 32) as u32;
        raw[10] = c.cdw10;
        raw[11] = c.cdw11;
        raw[12] = c.cdw12;
        // SAFETY: slot `sq_tail` of our own submission queue page.
        unsafe {
            let slot = self.sq.add(usize::from(self.sq_tail) * 64).cast::<u32>();
            for (i, w) in raw.iter().enumerate() {
                write_volatile(slot.add(i), *w);
            }
        }
        self.sq_tail = (self.sq_tail + 1) % self.depth;
        fence(SeqCst);
        // SAFETY: doorbell register of this queue inside the mapped BAR.
        unsafe { write_volatile(self.sq_db, u32::from(self.sq_tail)) };

        let start = time::now_ns();
        loop {
            // SAFETY: slot `cq_head` of our completion queue page.
            let (dw0, dw2, dw3) = unsafe {
                let e = self.cq.add(usize::from(self.cq_head) * 16).cast::<u32>();
                (
                    read_volatile(e),
                    read_volatile(e.add(2)),
                    read_volatile(e.add(3)),
                )
            };
            if (dw3 >> 16) & 1 == u32::from(self.phase) {
                let _ = dw2;
                self.cq_head += 1;
                if self.cq_head == self.depth {
                    self.cq_head = 0;
                    self.phase = !self.phase;
                }
                fence(SeqCst);
                // SAFETY: completion doorbell, as above.
                unsafe { write_volatile(self.cq_db, u32::from(self.cq_head)) };
                if (dw3 & 0xFFFF) as u16 != cid {
                    return Err("completion for an unexpected command");
                }
                return Ok(Completion {
                    result: dw0,
                    status: (dw3 >> 17) as u16,
                });
            }
            if time::now_ns() - start > TIMEOUT_NS {
                return Err("command timed out");
            }
            core::hint::spin_loop();
        }
    }
}

pub struct Nvme {
    regs: u64,
    queues: Vec<IrqMutex<Queue>>,
    nsid: u32,
    block_size: usize,
    blocks: u64,
    max_blocks: usize,
    pub model: String,
}

fn rd32(regs: u64, off: u64) -> u32 {
    // SAFETY: register inside the mapped BAR.
    unsafe { read_volatile((regs + off) as *const u32) }
}
fn wr32(regs: u64, off: u64, v: u32) {
    // SAFETY: as above.
    unsafe { write_volatile((regs + off) as *mut u32, v) }
}
fn rd64(regs: u64, off: u64) -> u64 {
    // SAFETY: as above.
    unsafe { read_volatile((regs + off) as *const u64) }
}
fn wr64(regs: u64, off: u64, v: u64) {
    // SAFETY: as above.
    unsafe { write_volatile((regs + off) as *mut u64, v) }
}

fn wait_ready(regs: u64, want: bool, timeout_ms: u64, what: &'static str) -> Result<(), InitError> {
    let start = time::now_ns();
    while (rd32(regs, REG_CSTS) & 1 == 1) != want {
        if time::now_ns() - start > timeout_ms * 1_000_000 {
            return Err(InitError::Timeout(what));
        }
        core::hint::spin_loop();
    }
    Ok(())
}

fn admin(q: &mut Queue, what: &'static str, c: Command) -> Result<Completion, InitError> {
    let done = q.execute(&c).map_err(InitError::Timeout)?;
    if done.status != 0 {
        return Err(InitError::Command(what, done.status));
    }
    Ok(done)
}

/// Physical address of a kernel heap or direct-map buffer.
fn dma_phys(p: *const u8) -> u64 {
    let va = p as u64;
    assert!(va >= mm::hhdm(), "DMA buffer is not in the direct map");
    va - mm::hhdm()
}

impl Nvme {
    pub fn init(dev: &Device) -> Result<Arc<Nvme>, InitError> {
        let bar = dev
            .bars
            .iter()
            .find(|b| b.index == 0)
            .ok_or(InitError::NoBar)?;
        pci::enable(dev.addr);
        // Interrupts are not used yet.
        let cmd = pci::read16(dev.addr, 4);
        pci::write16(dev.addr, 4, cmd | pci::CMD_INTX_DISABLE);
        let regs = paging::map_mmio(bar.phys, bar.size).map_err(|_| InitError::Mmio)?;

        let cap = rd64(regs, REG_CAP);
        let mqes = (cap & 0xFFFF) as u16;
        let timeout_ms = (((cap >> 24) & 0xFF) + 1) * 500;
        let dstrd = (cap >> 32) & 0xF;
        if cap & (1 << 37) == 0 {
            return Err(InitError::Unsupported("NVM command set"));
        }
        if (cap >> 48) & 0xF != 0 {
            return Err(InitError::Unsupported("4 KiB memory pages"));
        }
        let io_depth = IO_DEPTH.min(mqes + 1);

        // Reset, then program the admin queue pair.
        wr32(regs, REG_CC, 0);
        wait_ready(regs, false, timeout_ms, "controller disable")?;
        let mut aq = Queue::new(regs, dstrd, 0, ADMIN_DEPTH, false)?;
        wr32(
            regs,
            REG_AQA,
            u32::from(ADMIN_DEPTH - 1) << 16 | u32::from(ADMIN_DEPTH - 1),
        );
        wr64(regs, REG_ASQ, aq.sq_phys);
        wr64(regs, REG_ACQ, aq.cq_phys);
        // NVM command set, 4 KiB pages, round-robin, SQ entry 64 B, CQ entry 16 B.
        wr32(regs, REG_CC, (4 << 20) | (6 << 16) | 1);
        wait_ready(regs, true, timeout_ms, "controller enable")?;

        // Identify controller.
        let buf = frame::alloc_zeroed().ok_or(InitError::NoMemory)?;
        let id = mm::phys_to_virt(buf);
        admin(
            &mut aq,
            "identify controller",
            Command {
                opcode: 0x06,
                prp1: buf,
                cdw10: 1,
                ..Default::default()
            },
        )?;
        // SAFETY: `id` is a zeroed frame the controller just filled.
        let ident = unsafe { core::slice::from_raw_parts(id, 4096) };
        let model = String::from_utf8_lossy(&ident[24..64]).trim().into();
        let mdts = ident[77];
        let max_transfer = if mdts == 0 {
            MAX_TRANSFER
        } else {
            (4096usize << mdts).min(MAX_TRANSFER)
        };

        // Identify namespace 1.
        let nsid = 1u32;
        // SAFETY: reuse the same frame.
        unsafe { core::ptr::write_bytes(id, 0, 4096) };
        admin(
            &mut aq,
            "identify namespace",
            Command {
                opcode: 0x06,
                nsid,
                prp1: buf,
                cdw10: 0,
                ..Default::default()
            },
        )?;
        // SAFETY: as above.
        let ns = unsafe { core::slice::from_raw_parts(id, 4096) };
        let blocks = u64::from_le_bytes(ns[0..8].try_into().unwrap());
        let flbas = ns[26] & 0x0F;
        let lbaf = u32::from_le_bytes(
            ns[128 + 4 * usize::from(flbas)..128 + 4 * usize::from(flbas) + 4]
                .try_into()
                .unwrap(),
        );
        let lbads = (lbaf >> 16) & 0xFF;
        let metadata = lbaf & 0xFFFF;
        frame::free_frame(buf);
        if blocks == 0 {
            return Err(InitError::Unsupported("empty namespace"));
        }
        if metadata != 0 {
            return Err(InitError::Unsupported("per-block metadata"));
        }
        if !(9..=12).contains(&lbads) {
            return Err(InitError::Unsupported("block size"));
        }
        let block_size = 1usize << lbads;

        // Ask for one queue pair per CPU.
        let want = sched::online_cpus().max(1) as u32;
        let granted = admin(
            &mut aq,
            "set number of queues",
            Command {
                opcode: 0x09,
                cdw10: 0x07,
                cdw11: (want - 1) << 16 | (want - 1),
                ..Default::default()
            },
        )?
        .result;
        let n = (want.min((granted & 0xFFFF) + 1)).min((granted >> 16) + 1) as u16;

        let mut queues = Vec::new();
        for qid in 1..=n {
            let q = Queue::new(regs, dstrd, qid, io_depth, true)?;
            admin(
                &mut aq,
                "create I/O completion queue",
                Command {
                    opcode: 0x05,
                    prp1: q.cq_phys,
                    cdw10: u32::from(io_depth - 1) << 16 | u32::from(qid),
                    cdw11: 1, // physically contiguous, no interrupts
                    ..Default::default()
                },
            )?;
            admin(
                &mut aq,
                "create I/O submission queue",
                Command {
                    opcode: 0x01,
                    prp1: q.sq_phys,
                    cdw10: u32::from(io_depth - 1) << 16 | u32::from(qid),
                    cdw11: u32::from(qid) << 16 | 1,
                    ..Default::default()
                },
            )?;
            queues.push(IrqMutex::new(q));
        }
        Ok(Arc::new(Nvme {
            regs,
            queues,
            nsid,
            block_size,
            blocks,
            max_blocks: max_transfer / block_size,
            model,
        }))
    }

    pub fn queue_count(&self) -> usize {
        self.queues.len()
    }

    fn io(&self, opcode: u8, lba: u64, buf: *const u8, len: usize) -> Result<(), BlockError> {
        let nblocks = len / self.block_size;
        let mut q = self.queues[cpu_index() % self.queues.len()].lock();
        let start = dma_phys(buf);
        let first_len = (PAGE_SIZE - start % PAGE_SIZE) as usize;
        let (prp1, prp2) = if len <= first_len {
            (start, 0)
        } else {
            let rest = start + first_len as u64;
            if len - first_len <= PAGE_SIZE as usize {
                (start, rest)
            } else {
                let pages = (len - first_len).div_ceil(PAGE_SIZE as usize);
                for i in 0..pages {
                    // SAFETY: `pages` <= 512 entries fit in the queue's list page.
                    unsafe {
                        q.prp_list
                            .add(i)
                            .write_volatile(rest + i as u64 * PAGE_SIZE)
                    };
                }
                (start, q.prp_list_phys)
            }
        };
        let done = q
            .execute(&Command {
                opcode,
                nsid: self.nsid,
                prp1,
                prp2,
                cdw10: lba as u32,
                cdw11: (lba >> 32) as u32,
                cdw12: (nblocks - 1) as u32,
            })
            .map_err(|_| BlockError::Device)?;
        if done.status != 0 {
            return Err(BlockError::Device);
        }
        Ok(())
    }

    fn transfer(&self, opcode: u8, lba: u64, buf: *const u8, len: usize) -> Result<(), BlockError> {
        let bs = self.block_size;
        let chunk = self.max_blocks * bs;
        let mut done = 0usize;
        while done < len {
            let n = chunk.min(len - done);
            // SAFETY: stays inside the caller's buffer.
            self.io(
                opcode,
                lba + (done / bs) as u64,
                unsafe { buf.add(done) },
                n,
            )?;
            done += n;
        }
        Ok(())
    }
}

impl BlockDevice for Nvme {
    fn block_size(&self) -> usize {
        self.block_size
    }
    fn block_count(&self) -> u64 {
        self.blocks
    }
    fn read(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        block::check(self, lba, buf.len())?;
        self.transfer(0x02, lba, buf.as_ptr(), buf.len())
    }
    fn write(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        block::check(self, lba, buf.len())?;
        self.transfer(0x01, lba, buf.as_ptr(), buf.len())
    }
    fn flush(&self) -> Result<(), BlockError> {
        let mut q = self.queues[cpu_index() % self.queues.len()].lock();
        let done = q
            .execute(&Command {
                opcode: 0x00,
                nsid: self.nsid,
                ..Default::default()
            })
            .map_err(|_| BlockError::Device)?;
        if done.status != 0 {
            return Err(BlockError::Device);
        }
        Ok(())
    }
}

/// Bring up every NVMe controller; returns how many block devices appeared.
pub fn init() -> usize {
    let mut n = 0;
    for dev in pci::devices()
        .iter()
        .filter(|d| d.class == 0x01 && d.subclass == 0x08 && d.prog_if == 0x02)
    {
        match Nvme::init(dev) {
            Ok(c) => {
                let name = format!("nvme{n}n1");
                kprintln!(
                    "[nvme] {} {}: {} MiB, {} B blocks, {} I/O queues, {}",
                    dev.addr,
                    name,
                    c.blocks * c.block_size as u64 / (1 << 20),
                    c.block_size,
                    c.queue_count(),
                    c.model
                );
                block::register(&name, c);
                n += 1;
            }
            Err(e) => kprintln!("[nvme] {}: init failed: {e:?}", dev.addr),
        }
    }
    n
}
