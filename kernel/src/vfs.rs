//! Virtual file system: a mount table, path resolution and the common
//! operations every filesystem driver implements.

#![allow(dead_code)]

use alloc::{string::String, sync::Arc, vec::Vec};
use spin::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsError {
    NotFound,
    NotDir,
    IsDir,
    /// On-disk structures are inconsistent; the driver refuses to guess.
    Corrupt(&'static str),
    Io,
    Unsupported(&'static str),
    NameTooLong,
    Loop,
    NoSpace,
    Exists,
    NotEmpty,
    ReadOnly,
    Invalid,
}

impl From<crate::block::BlockError> for FsError {
    fn from(_: crate::block::BlockError) -> Self {
        FsError::Io
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    File,
    Dir,
    Symlink,
}

/// A resolved file or directory. `data` is private to the driver that made it.
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub kind: Kind,
    pub size: u64,
    pub data: [u64; 3],
}

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub node: Node,
}

pub trait FileSystem: Send + Sync {
    fn root(&self) -> Node;
    fn lookup(&self, dir: &Node, name: &str) -> Result<Node, FsError>;
    fn read(&self, file: &Node, offset: u64, buf: &mut [u8]) -> Result<usize, FsError>;
    fn read_dir(&self, dir: &Node) -> Result<Vec<DirEntry>, FsError>;
    fn read_link(&self, _link: &Node) -> Result<String, FsError> {
        Err(FsError::Unsupported("symbolic links"))
    }
    fn create(&self, _dir: &Node, _name: &str, _kind: Kind) -> Result<Node, FsError> {
        Err(FsError::ReadOnly)
    }
    fn write(&self, _file: &Node, _offset: u64, _data: &[u8]) -> Result<usize, FsError> {
        Err(FsError::ReadOnly)
    }
    fn truncate(&self, _file: &Node, _size: u64) -> Result<Node, FsError> {
        Err(FsError::ReadOnly)
    }
    fn remove(&self, _dir: &Node, _name: &str) -> Result<(), FsError> {
        Err(FsError::ReadOnly)
    }
    /// Re-read a node's current attributes (size changes after writes).
    fn refresh(&self, node: &Node) -> Result<Node, FsError> {
        Ok(*node)
    }
    fn sync(&self) -> Result<(), FsError> {
        Ok(())
    }
}

struct Mount {
    path: String,
    fs: Arc<dyn FileSystem>,
}

static MOUNTS: RwLock<Vec<Mount>> = RwLock::new(Vec::new());

const MAX_SYMLINKS: usize = 16;
pub const NAME_MAX: usize = 255;

fn normalize(path: &str) -> Result<Vec<&str>, FsError> {
    if !path.starts_with('/') {
        return Err(FsError::Invalid);
    }
    let mut out: Vec<&str> = Vec::new();
    for c in path.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            c if c.len() > NAME_MAX => return Err(FsError::NameTooLong),
            c => out.push(c),
        }
    }
    Ok(out)
}

pub fn mount(path: &str, fs: Arc<dyn FileSystem>) -> Result<(), FsError> {
    let comps = normalize(path)?;
    let canon = alloc::format!("/{}", comps.join("/"));
    let mut m = MOUNTS.write();
    if m.iter().any(|x| x.path == canon) {
        return Err(FsError::Exists);
    }
    m.push(Mount { path: canon, fs });
    Ok(())
}

pub fn unmount_all() {
    MOUNTS.write().clear();
}

/// The mount owning `comps` and how many leading components it consumes.
fn mount_for(comps: &[&str]) -> Option<(Arc<dyn FileSystem>, usize)> {
    let m = MOUNTS.read();
    let mut best: Option<(&Mount, usize)> = None;
    for x in m.iter() {
        let mc: Vec<&str> = x.path.split('/').filter(|s| !s.is_empty()).collect();
        if comps.len() >= mc.len()
            && comps[..mc.len()] == mc[..]
            && best.as_ref().is_none_or(|(_, n)| mc.len() > *n)
        {
            best = Some((x, mc.len()));
        }
    }
    best.map(|(x, n)| (x.fs.clone(), n))
}

/// A node together with the filesystem it belongs to.
#[derive(Clone)]
pub struct Resolved {
    pub fs: Arc<dyn FileSystem>,
    pub node: Node,
}

fn resolve_inner(path: &str, follow_last: bool, depth: usize) -> Result<Resolved, FsError> {
    if depth > MAX_SYMLINKS {
        return Err(FsError::Loop);
    }
    let comps = normalize(path)?;
    let (fs, used) = mount_for(&comps).ok_or(FsError::NotFound)?;
    let mut cur = fs.root();
    let mut walked: Vec<&str> = comps[..used].to_vec();
    let rest = &comps[used..];
    for (i, name) in rest.iter().enumerate() {
        if cur.kind != Kind::Dir {
            return Err(FsError::NotDir);
        }
        let mut next = fs.lookup(&cur, name)?;
        let last = i + 1 == rest.len();
        if next.kind == Kind::Symlink && (!last || follow_last) {
            let target = fs.read_link(&next)?;
            let base = alloc::format!("/{}", walked.join("/"));
            let joined = if target.starts_with('/') {
                target
            } else {
                alloc::format!("{base}/{target}")
            };
            let remaining = rest[i + 1..].join("/");
            let full = if remaining.is_empty() {
                joined
            } else {
                alloc::format!("{joined}/{remaining}")
            };
            return resolve_inner(&full, follow_last, depth + 1);
        }
        walked.push(name);
        if last {
            next = fs.refresh(&next)?;
        }
        cur = next;
    }
    Ok(Resolved { fs, node: cur })
}

pub fn resolve(path: &str) -> Result<Resolved, FsError> {
    resolve_inner(path, true, 0)
}

pub fn resolve_nofollow(path: &str) -> Result<Resolved, FsError> {
    resolve_inner(path, false, 0)
}

pub fn stat(path: &str) -> Result<Node, FsError> {
    Ok(resolve(path)?.node)
}

pub fn read_dir(path: &str) -> Result<Vec<DirEntry>, FsError> {
    let r = resolve(path)?;
    if r.node.kind != Kind::Dir {
        return Err(FsError::NotDir);
    }
    r.fs.read_dir(&r.node)
}

pub fn read_at(path: &str, offset: u64, buf: &mut [u8]) -> Result<usize, FsError> {
    let r = resolve(path)?;
    if r.node.kind == Kind::Dir {
        return Err(FsError::IsDir);
    }
    r.fs.read(&r.node, offset, buf)
}

/// Whole file into memory (for configuration, programs; not for games).
pub fn read_file(path: &str) -> Result<Vec<u8>, FsError> {
    let r = resolve(path)?;
    if r.node.kind == Kind::Dir {
        return Err(FsError::IsDir);
    }
    let size = usize::try_from(r.node.size).map_err(|_| FsError::Unsupported("file too large"))?;
    let mut v = alloc::vec![0u8; size];
    let mut done = 0;
    while done < size {
        let n = r.fs.read(&r.node, done as u64, &mut v[done..])?;
        if n == 0 {
            return Err(FsError::Corrupt("file shorter than its recorded size"));
        }
        done += n;
    }
    Ok(v)
}

fn split_parent(path: &str) -> Result<(String, String), FsError> {
    let comps = normalize(path)?;
    let (name, dir) = comps.split_last().ok_or(FsError::Invalid)?;
    Ok((alloc::format!("/{}", dir.join("/")), (*name).into()))
}

pub fn create(path: &str, kind: Kind) -> Result<Node, FsError> {
    let (dir, name) = split_parent(path)?;
    let d = resolve(&dir)?;
    if d.node.kind != Kind::Dir {
        return Err(FsError::NotDir);
    }
    d.fs.create(&d.node, &name, kind)
}

pub fn write_at(path: &str, offset: u64, data: &[u8]) -> Result<usize, FsError> {
    let r = resolve(path)?;
    if r.node.kind == Kind::Dir {
        return Err(FsError::IsDir);
    }
    r.fs.write(&r.node, offset, data)
}

pub fn truncate(path: &str, size: u64) -> Result<(), FsError> {
    let r = resolve(path)?;
    r.fs.truncate(&r.node, size).map(|_| ())
}

pub fn remove(path: &str) -> Result<(), FsError> {
    let (dir, name) = split_parent(path)?;
    let d = resolve(&dir)?;
    d.fs.remove(&d.node, &name)
}

pub fn sync_all() -> Result<(), FsError> {
    let fss: Vec<_> = MOUNTS.read().iter().map(|m| m.fs.clone()).collect();
    for fs in fss {
        fs.sync()?;
    }
    Ok(())
}
