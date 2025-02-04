use core::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicI64, AtomicPtr, AtomicU64, AtomicUsize};

// ------------------------------------------------------
// Platform specific values
// ------------------------------------------------------

// ------------------------------------------------------
// Size of a pointer.
// We assume that `sizeof(void*)==sizeof(intptr_t)`
// and it holds for all platforms we know of.
//
// However, the C standard only requires that:
//  p == (void*)((intptr_t)p))
// but we also need:
//  i == (intptr_t)((void*)i)
// or otherwise one might define an intptr_t type that is larger than a pointer...
// ------------------------------------------------------

use crate::MI_SMALL_WSIZE_MAX;

pub(crate) const MI_INTPTR_SHIFT: usize = 3; // 8 bytes pointers on 64-bit
pub(crate) const MI_INTPTR_SIZE: usize = 1 << MI_INTPTR_SHIFT;

// ------------------------------------------------------
// Main internal data-structures
// ------------------------------------------------------

// Main tuning parameters for segment and page sizes
// Sizes for 64-bit, divide by two for 32-bit
pub(crate) const MI_SMALL_PAGE_SHIFT: usize = 13 + MI_INTPTR_SHIFT;
pub(crate) const MI_LARGE_PAGE_SHIFT: usize = 6 + MI_SMALL_PAGE_SHIFT;
pub(crate) const MI_SEGMENT_SHIFT: usize = MI_LARGE_PAGE_SHIFT;

// Derived constants
pub(crate) const MI_SEGMENT_SIZE: usize = 1 << MI_SEGMENT_SHIFT;
pub(crate) const MI_SEGMENT_MASK: usize = MI_SEGMENT_SIZE - 1;

pub(crate) const MI_SMALL_PAGE_SIZE: usize = 1 << MI_SMALL_PAGE_SHIFT;
pub(crate) const MI_LARGE_PAGE_SIZE: usize = 1 << MI_LARGE_PAGE_SHIFT;

pub(crate) const MI_SMALL_PAGES_PER_SEGMENT: usize = MI_SEGMENT_SIZE / MI_SMALL_PAGE_SIZE;
pub(crate) const MI_LARGE_PAGES_PER_SEGMENT: usize = MI_SEGMENT_SIZE / MI_LARGE_PAGE_SIZE;

pub(crate) const MI_LARGE_SIZE_MAX: usize = MI_LARGE_PAGE_SIZE / 8;
pub(crate) const MI_LARGE_WSIZE_MAX: usize = MI_LARGE_SIZE_MAX >> MI_INTPTR_SHIFT;

// Maximum number of size classes. (spaced exponentially in 16.7% increments)
pub(crate) const MI_BIN_HUGE: u8 = 64;

// Minimal alignment necessary. On most platforms 16 bytes are needed
// due to SSE registers for example. This must be at least `MI_INTPTR_SIZE`
pub(crate) const MI_MAX_ALIGN_SIZE: usize = 16; // sizeof(max_align_t)

// #if (MI_LARGE_WSIZE_MAX > 131072)
// #error "define more bins"
// #endif

pub(crate) type MiEncoded = usize;

// free lists contain blocks
#[repr(C, align(1))]
pub(crate) struct MiBlock {
    pub(crate) next: MiEncoded,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum MiDelayed {
    MiNoDelayedFree = 0,
    MiUseDelayedFree,
    MiDelayedFreeing,
}

// #[repr(C)]
// pub(crate) struct MiPageFlags {
//     // pub(crate) value: u16,
//     pub(crate) has_aligned: bool,
//     pub(crate) in_full: bool,
// }
pub(crate) struct MiPageFlags(pub u16);

impl MiPageFlags {
    pub(crate) const fn new() -> Self {
        MiPageFlags(0)
    }

    pub(crate) fn has_aligned(&self) -> bool {
        self.0 & 0b1 != 0
    }

    pub(crate) fn set_has_aligned(&mut self, has_aligned: bool) {
        if has_aligned {
            self.0 |= 0b1;
        } else {
            self.0 &= !0b1;
        }
    }

    pub(crate) fn in_full(&self) -> bool {
        self.0 & 0b10 != 0
    }

    pub(crate) fn set_in_full(&mut self, in_full: bool) {
        if in_full {
            self.0 |= 0b10;
        } else {
            self.0 &= !0b10;
        }
    }
}

// Thread free list.
// We use 2 bits of the pointer for the `use_delayed_free` and `delayed_freeing` flags.
//   typedef union mi_thread_free_u {
//     uintptr_t value;
//     struct {
//       mi_delayed_t delayed:2;
//   #if MI_INTPTR_SIZE==8
//       uintptr_t head:62;    // head free block in the list (right-shifted by 2)
//   #elif MI_INTPTR_SIZE==4
//       uintptr_t head:30;
//   #endif
//     };
//   } mi_thread_free_t;
// #[repr(C)]
// pub(crate) struct MiThreadFree {
//     pub(crate) value: AtomicUsize,
//     pub(crate) delayed: MiDelayed,
//     pub(crate) head: usize,
// }
pub(crate) struct MiThreadFree(AtomicUsize);

impl MiThreadFree {
    pub(crate) const fn new() -> Self {
        MiThreadFree(AtomicUsize::new(0))
    }

    pub(crate) fn delayed(&self) -> MiDelayed {
        match self.0.load(std::sync::atomic::Ordering::Relaxed) & 0b11 {
            0 => MiDelayed::MiNoDelayedFree,
            1 => MiDelayed::MiUseDelayedFree,
            2 => MiDelayed::MiDelayedFreeing,
            _ => unreachable!(),
        }
    }

    pub(crate) fn set_delayed(&self, delayed: MiDelayed) {
        let mut value = self.0.load(std::sync::atomic::Ordering::Relaxed);
        value = (value & !0b11) | delayed as usize;
        self.0.store(value, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn head(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed) >> 2
    }

    pub(crate) fn set_head(&self, head: usize) {
        let mut value = self.0.load(std::sync::atomic::Ordering::Relaxed);
        value = (value & 0b11) | (head << 2);
        self.0.store(value, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn load(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn store(&self, value: usize) {
        self.0.store(value, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn compare_exchange(&self, current: usize, new: usize) -> Result<usize, usize> {
        self.0.compare_exchange(
            current,
            new,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Relaxed,
        )
    }
}

impl Clone for MiThreadFree {
    fn clone(&self) -> Self {
        MiThreadFree(AtomicUsize::new(
            self.0.load(std::sync::atomic::Ordering::Relaxed),
        ))
    }
}

pub(crate) const MI_TF_PTR_SHIFT: usize = 2;

// A page contains blocks of one specific size (`block_size`).
// Each page has three list of free blocks:
// `free` for blocks that can be allocated,
// `local_free` for freed blocks that are not yet available to `mi_malloc`
// `thread_free` for freed blocks by other threads
// The `local_free` and `thread_free` lists are migrated to the `free` list
// when it is exhausted. The separate `local_free` list is necessary to
// implement a monotonic heartbeat. The `thread_free` list is needed for
// avoiding atomic operations in the common case.
//
// `used - thread_freed` == actual blocks that are in use (alive)
// `used - thread_freed + |free| + |local_free| == capacity`
//
// note: we don't count `freed` (as |free|) instead of `used` to reduce
//       the number of memory accesses in the `mi_page_all_free` function(s).
// note: the funny layout here is due to:
// - access is optimized for `mi_free` and `mi_page_alloc`
// - using `uint16_t` does not seem to slow things down
#[repr(C)]
pub(crate) struct MiPage {
    // "owned" by the segment
    pub(crate) segment_idx: u8, // index in the segment `pages` array, `page == &segment->pages[page->segment_idx]`
    pub(crate) segment_in_use: bool, // `true` if the segment allocated this page
    pub(crate) is_reset: bool,  // `true` if the page memory was reset

    // layout like this to optimize access in `mi_malloc` and `mi_free`
    pub(crate) flags: MiPageFlags,
    pub(crate) capacity: u16, // number of blocks committed
    pub(crate) reserved: u16, // number of blocks reserved in memory

    pub(crate) free: *mut MiBlock, // list of available free blocks (`malloc` allocates from this list)
    pub(crate) used: usize, // number of blocks in use (including blocks in `local_free` and `thread_free`)

    pub(crate) local_free: *mut MiBlock, // list of deferred free blocks by this thread (migrates to `free`)
    pub(crate) thread_freed: AtomicUsize, // (volatile) at least this number of blocks are in `thread_free`
    pub(crate) thread_free: MiThreadFree, // (volatile) list of deferred free blocks freed by other threads

    // less accessed info
    pub(crate) block_size: usize, // size available in each block (always `>0`)
    pub(crate) heap: *mut MiHeap, // the owning heap
    pub(crate) next: *mut MiPage, // next page owned by this thread with the same `block_size`
    pub(crate) prev: *mut MiPage, // previous page owned by this thread with the same `block_size`

                                  // improve page index calculation
                                  //   #if MI_INTPTR_SIZE==8
                                  //     //void*                 padding[1];        // 10 words on 64-bit
                                  //   #elif MI_INTPTR_SIZE==4
                                  //     void*                 padding[1];          // 12 words on 32-bit
                                  //   #endif
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MiPageKind {
    MiPageSmall, // small blocks go into 64kb pages inside a segment
    MiPageLarge, // larger blocks go into a single page spanning a whole segment
    MiPageHuge, // huge blocks (>512kb) are put into a single page in a segment of the exact size (but still 2mb aligned)
}

impl Display for MiPageKind {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            MiPageKind::MiPageSmall => write!(f, "MI_PAGE_SMALL"),
            MiPageKind::MiPageLarge => write!(f, "MI_PAGE_LARGE"),
            MiPageKind::MiPageHuge => write!(f, "MI_PAGE_HUGE"),
        }
    }
}

// Segments are large allocated memory blocks (2mb on 64 bit) from
// the OS. Inside segments we allocated fixed size _pages_ that
// contain blocks.
#[repr(C)]
pub(crate) struct MiSegment {
    pub(crate) next: *mut MiSegment,
    pub(crate) prev: *mut MiSegment,
    pub(crate) abandoned_next: *mut MiSegment,
    pub(crate) abandoned: usize, // abandoned pages (i.e. the original owning thread stopped) (`abandoned <= used`)
    pub(crate) used: usize,      // count of pages in use (`used <= capacity`)
    pub(crate) capacity: usize,  // count of available pages (`#free + used`)
    pub(crate) segment_size: usize, // for huge pages this may be different from `MI_SEGMENT_SIZE`
    pub(crate) segment_info_size: usize, // space we are using from the first page for segment meta-data and possible guard pages.

    // layout like this to optimize access in `mi_free`
    pub(crate) page_shift: usize, // `1 << page_shift` == the page sizes == `page->block_size * page->reserved` (unless the first page, then `-segment_info_size`).
    pub(crate) thread_id: usize,  // unique id of the thread owning this segment
    pub(crate) page_kind: MiPageKind, // kind of pages: small, large, or huge
    pub(crate) page0: MiPage,     // up to `MI_SMALL_PAGES_PER_SEGMENT` pages
}

impl MiSegment {
    pub(crate) fn pages(&self) -> *mut MiPage {
        &self.page0 as *const MiPage as *mut MiPage
    }
}

// ------------------------------------------------------
// Heaps
// Provide first-class heaps to allocate from.
// A heap just owns a set of pages for allocation and
// can only be allocate/reallocate from the thread that created it.
// Freeing blocks can be done from any thread though.
// Per thread, the segments are shared among its heaps.
// Per thread, there is always a default heap that is
// used for allocation; it is initialized to statically
// point to an empty heap to avoid initialization checks
// in the fast path.
// ------------------------------------------------------

// Pages of a certain block size are held in a queue.
#[repr(C)]
pub(crate) struct MiPageQueue {
    pub(crate) first: *mut MiPage,
    pub(crate) last: *mut MiPage,
    pub(crate) block_size: usize,
}

pub(crate) const MI_BIN_FULL: u8 = MI_BIN_HUGE + 1;

// A heap owns a set of pages.
#[repr(C)]
pub(crate) struct MiHeap {
    pub(crate) tld: *mut MiTld,
    pub(crate) pages_free_direct: [*mut MiPage; MI_SMALL_WSIZE_MAX + 2], // optimize: array where every entry points a page with possibly free blocks in the corresponding queue for that size.
    pub(crate) pages: [MiPageQueue; (MI_BIN_FULL + 1) as usize], // queue of pages for each size class (or "bin")
    pub(crate) thread_delayed_free: AtomicPtr<MiBlock>,
    pub(crate) thread_id: usize,  // thread this heap belongs too
    pub(crate) page_count: usize, // total number of pages in the `pages` queues.
    pub(crate) no_reclaim: bool,  // `true` if this heap should not reclaim abandoned pages
}

unsafe impl Sync for MiHeap {}

// ------------------------------------------------------
// Debug
// ------------------------------------------------------
#[track_caller]
#[inline]
pub fn mi_assert(cond: bool) {
    assert!(cond);
}

#[track_caller]
#[inline]
pub fn mi_assert_internal(cond: bool) {
    assert!(cond);
}

#[track_caller]
#[inline]
pub fn mi_assert_expensive(cond: bool) {
    assert!(cond);
}

// ------------------------------------------------------
// Statistics
// ------------------------------------------------------

#[derive(Debug, Default)]
pub(crate) struct MiStatCount {
    pub(crate) allocated: AtomicI64,
    pub(crate) freed: AtomicI64,
    pub(crate) peak: AtomicI64,
    pub(crate) current: AtomicI64,
}

#[derive(Debug, Default)]
pub(crate) struct MiStatCounter {
    pub(crate) total: AtomicI64,
    pub(crate) count: AtomicI64,
}

#[derive(Debug)]
pub(crate) struct MiStats {
    pub(crate) segments: MiStatCount,
    pub(crate) pages: MiStatCount,
    pub(crate) reserved: MiStatCount,
    pub(crate) committed: MiStatCount,
    pub(crate) reset: MiStatCount,
    pub(crate) segments_abandoned: MiStatCount,
    pub(crate) pages_abandoned: MiStatCount,
    pub(crate) pages_extended: MiStatCount,
    pub(crate) mmap_calls: MiStatCount,
    pub(crate) mmap_right_align: MiStatCount,
    pub(crate) mmap_ensure_aligned: MiStatCount,
    pub(crate) threads: MiStatCount,
    pub(crate) huge: MiStatCount,
    pub(crate) malloc: MiStatCount,
    pub(crate) searches: MiStatCounter,
    pub(crate) normal: [MiStatCount; (MI_BIN_HUGE + 1) as usize],
}

impl Default for MiStats {
    fn default() -> Self {
        MiStats {
            segments: MiStatCount::default(),
            pages: MiStatCount::default(),
            reserved: MiStatCount::default(),
            committed: MiStatCount::default(),
            reset: MiStatCount::default(),
            segments_abandoned: MiStatCount::default(),
            pages_abandoned: MiStatCount::default(),
            pages_extended: MiStatCount::default(),
            mmap_calls: MiStatCount::default(),
            mmap_right_align: MiStatCount::default(),
            mmap_ensure_aligned: MiStatCount::default(),
            threads: MiStatCount::default(),
            huge: MiStatCount::default(),
            malloc: MiStatCount::default(),
            searches: MiStatCounter::default(),
            normal: std::array::from_fn(|_| MiStatCount::default()),
        }
    }
}

/// ------------------------------------------------------
/// Thread Local data
/// ------------------------------------------------------

// Queue of segments
pub(crate) struct MiSegmentQueue {
    pub(crate) first: *mut MiSegment,
    pub(crate) last: *mut MiSegment,
}

// Segments thread local data
pub(crate) struct MiSegmentsTld {
    pub(crate) small_free: MiSegmentQueue, // queue of segments with free small pages
    pub(crate) current_size: usize,        // current size of all segments
    pub(crate) peak_size: usize,           // peak size of all segments
    pub(crate) cache_count: usize,         // number of segments in the cache
    pub(crate) cache_size: usize,          // total size of all segments in the cache
    pub(crate) cache: MiSegmentQueue, // (small) cache of segments for small and large pages (to avoid repeated mmap calls)
                                      // pub(crate) stats: *mut MiStats,   // points to tld stats
}

// OS thread local data
pub(crate) struct MiOsTld {
    pub(crate) mmap_next_probable: usize, // probable next address start allocated by mmap (to guess which path to take on alignment)
    pub(crate) mmap_previous: *mut u8,    // previous address returned by mmap
    pub(crate) pool: *mut u8,             // pool of segments to reduce mmap calls on some platforms
    pub(crate) pool_available: usize,     // bytes available in the pool
                                          // pub(crate) stats: *mut MiStats,       // points to tld stats
}

// Thread local data
pub(crate) struct MiTld {
    pub(crate) heartbeat: AtomicU64,      // monotonic heartbeat count
    pub(crate) heap_backing: *mut MiHeap, // backing heap of this thread (cannot be deleted)
    pub(crate) segments: MiSegmentsTld,   // segment tld
    pub(crate) os: MiOsTld,               // os tld
                                          // pub(crate) stats: MiStats,            // statistics
}

unsafe impl Sync for MiTld {}
