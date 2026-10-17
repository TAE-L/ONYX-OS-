//! GDT and TSS.
//!
//! Segment order is fixed by the SYSCALL/SYSRET contract used in Phase 3:
//! kernel code (0x08), kernel data (0x10), user data (0x18), user code (0x20).
//! The TSS gives double faults and NMIs their own stacks (IST1, IST2) so they
//! cannot be taken on a corrupt stack.

use core::cell::UnsafeCell;
use spin::Once;
use x86_64::VirtAddr;
use x86_64::instructions::segmentation::{CS, DS, ES, SS, Segment};
use x86_64::instructions::tables::load_tss;
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
use x86_64::structures::tss::TaskStateSegment;

const IST_STACK_SIZE: usize = 16 * 1024;
pub const DOUBLE_FAULT_IST: u8 = 1;
pub const NMI_IST: u8 = 2;

#[repr(align(16))]
struct IstStack(UnsafeCell<[u8; IST_STACK_SIZE]>);

// SAFETY: only ever used as raw stack memory by the CPU.
unsafe impl Sync for IstStack {}

static DOUBLE_FAULT_STACK: IstStack = IstStack(UnsafeCell::new([0; IST_STACK_SIZE]));
static NMI_STACK: IstStack = IstStack(UnsafeCell::new([0; IST_STACK_SIZE]));

fn stack_top(stack: &'static IstStack) -> VirtAddr {
    VirtAddr::from_ptr(stack.0.get()) + IST_STACK_SIZE as u64
}

pub struct Selectors {
    pub kernel_code: SegmentSelector,
    pub kernel_data: SegmentSelector,
    // Consumed by the Phase 3 syscall/ring-3 entry code.
    #[allow(dead_code)]
    pub user_data: SegmentSelector,
    pub user_code: SegmentSelector,
    tss: SegmentSelector,
}

static TSS: Once<TaskStateSegment> = Once::new();
static GDT: Once<(GlobalDescriptorTable, Selectors)> = Once::new();

pub fn selectors() -> &'static Selectors {
    &GDT.get().expect("GDT not initialised").1
}

pub fn init() {
    let tss = TSS.call_once(|| {
        let mut tss = TaskStateSegment::new();
        tss.interrupt_stack_table[usize::from(DOUBLE_FAULT_IST) - 1] =
            stack_top(&DOUBLE_FAULT_STACK);
        tss.interrupt_stack_table[usize::from(NMI_IST) - 1] = stack_top(&NMI_STACK);
        tss
    });
    let (gdt, sel) = GDT.call_once(|| {
        let mut gdt = GlobalDescriptorTable::new();
        let kernel_code = gdt.append(Descriptor::kernel_code_segment());
        let kernel_data = gdt.append(Descriptor::kernel_data_segment());
        let user_data = gdt.append(Descriptor::user_data_segment());
        let user_code = gdt.append(Descriptor::user_code_segment());
        let tss_sel = gdt.append(Descriptor::tss_segment(tss));
        (
            gdt,
            Selectors {
                kernel_code,
                kernel_data,
                user_data,
                user_code,
                tss: tss_sel,
            },
        )
    });
    assert_eq!(sel.kernel_code.0, 0x08);
    assert_eq!(sel.user_code.0, 0x23);

    // SAFETY: the GDT is 'static and its selectors match the descriptors above.
    unsafe {
        gdt.load();
        CS::set_reg(sel.kernel_code);
        SS::set_reg(sel.kernel_data);
        DS::set_reg(sel.kernel_data);
        ES::set_reg(sel.kernel_data);
        load_tss(sel.tss);
    }
    kprintln!("[gdt] loaded; double-fault and NMI run on dedicated IST stacks");
}
