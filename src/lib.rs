mod alloc;
mod heap;
mod init;
mod internal;
mod os;
mod page;
mod page_queue;
mod segment;
mod types;

// ------------------------------------------------------
// Standard malloc interface
// ------------------------------------------------------

pub use alloc::mi_free;
pub use alloc::mi_malloc;

// ------------------------------------------------------
// Extended functionality
// ------------------------------------------------------

const MI_SMALL_WSIZE_MAX: usize = 128;
const MI_SMALL_SIZE_MAX: usize = MI_SMALL_WSIZE_MAX * std::mem::size_of::<usize>();

// ------------------------------------------------------
// Aligned allocation
// ------------------------------------------------------

// ------------------------------------------------------
// Heaps
// ------------------------------------------------------

// ------------------------------------------------------
// Analysis
// ------------------------------------------------------

// An area of heap space contains blocks of a single size.
pub(crate) struct MiHeapArea {
    blocks: *mut u8,   // start of the area containing heap blocks
    reserved: usize,   // bytes reserved for this area (virtual)
    committed: usize,  // current available bytes for this area
    used: usize,       // bytes in use by allocated blocks
    block_size: usize, // size in bytes of each block
}

pub(crate) type MiBlockVisitFun = fn(
    heap: *mut MiHeap,
    area: *mut MiHeapArea,
    block: *mut u8,
    block_size: usize,
    arg: *mut u8,
) -> bool;

// ------------------------------------------------------
// Convenience
// ------------------------------------------------------
#[macro_export]
macro_rules! mi_malloc_tp {
    ($tp:ty) => {
        mi_malloc(std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_zalloc_tp {
    ($tp:ty) => {
        mi_zalloc(std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_calloc_tp {
    ($tp:ty, $n:expr) => {
        mi_calloc($n, std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_mallocn_tp {
    ($tp:ty, $n:expr) => {
        mi_mallocn($n, std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_reallocn_tp {
    ($p:expr, $tp:ty, $n:expr) => {
        mi_reallocn($p, $n, std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_recalloc_tp {
    ($p:expr, $tp:ty, $n:expr) => {
        mi_recalloc($p, $n, std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_heap_malloc_tp {
    ($hp:expr, $tp:ty) => {
        mi_heap_malloc($hp, std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_heap_zalloc_tp {
    ($hp:expr, $tp:ty) => {
        mi_heap_zalloc($hp, std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_heap_calloc_tp {
    ($hp:expr, $tp:ty, $n:expr) => {
        mi_heap_calloc($hp, $n, std::mem::size_of::<$tp>()) as *mut $tp
    };
}

#[macro_export]
macro_rules! mi_heap_mallocn_tp {
    ($hp:expr, $tp:ty, $n:expr) => {
        mi_heap_mallocn($hp, $n, std::mem::size_of::<$tp>()) as *mut $tp
    };
}
use types::MiHeap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malloc_and_free() {
        for i in 1..100000 {
            let n = unsafe { libc::rand() as usize % i + 1 };
            let buf = unsafe { mi_malloc(n) };

            // write to the memory to make sure it is actually allocated
            let arr = unsafe { std::slice::from_raw_parts_mut(buf, n) };
            arr[0] = n as u8;
            arr[(n - 1) / 2] = n as u8;
            arr[n - 1] = n as u8;

            unsafe { mi_free(buf) };
        }
    }
}
