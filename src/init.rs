use crate::heap::{mi_heap_collect_abandon, mi_heap_set_default};
use crate::internal::{mi_get_default_heap, mi_heap_is_initialized, mi_thread_id, MI_INIT};
use crate::os::{mi_os_alloc, mi_os_free};
use crate::types::MI_LARGE_WSIZE_MAX;
use crate::types::{
    MiHeap, MiOsTld, MiPage, MiPageFlags, MiPageQueue, MiSegmentQueue, MiSegmentsTld, MiThreadFree,
    MiTld, MI_INTPTR_SIZE,
};
use ctor::{ctor, dtor};
use libc::{pthread_key_create, pthread_key_t, pthread_setspecific};
use std::cell::Cell;
use std::ptr::{addr_of, addr_of_mut, null_mut};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize};
use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};

// Empty page used to initialize the small free pages array
// mut is useless here
pub(crate) static mut MI_PAGE_EMPTY: MiPage = MiPage {
    segment_idx: 0,
    segment_in_use: false,
    is_reset: false,
    flags: MiPageFlags::new(),
    capacity: 0,
    reserved: 0,
    free: null_mut(),
    used: 0,
    local_free: null_mut(),
    thread_freed: AtomicUsize::new(0),
    thread_free: MiThreadFree::new(),
    block_size: 0,
    heap: null_mut(),
    next: null_mut(),
    prev: null_mut(),
};

macro_rules! MI_SMALL_PAGES_EMPTY {
    () => {
        unsafe { MI_INIT![&MI_PAGE_EMPTY as *const MiPage as *mut MiPage; 130] }
    };
}

macro_rules! QNULL {
    ($n:expr) => {
        MiPageQueue {
            first: null_mut(),
            last: null_mut(),
            block_size: $n * std::mem::size_of::<usize>(),
        }
    };
}

macro_rules! MI_PAGE_QUEUES_EMPTY {
    () => {
        [
            QNULL!(1),
            QNULL!(1),
            QNULL!(2),
            QNULL!(3),
            QNULL!(4),
            QNULL!(5),
            QNULL!(6),
            QNULL!(7),
            QNULL!(8),
            QNULL!(10),
            QNULL!(12),
            QNULL!(14),
            QNULL!(16),
            QNULL!(20),
            QNULL!(24),
            QNULL!(28),
            QNULL!(32),
            QNULL!(40),
            QNULL!(48),
            QNULL!(56),
            QNULL!(64),
            QNULL!(80),
            QNULL!(96),
            QNULL!(112),
            QNULL!(128),
            QNULL!(160),
            QNULL!(192),
            QNULL!(224),
            QNULL!(256),
            QNULL!(320),
            QNULL!(384),
            QNULL!(448),
            QNULL!(512),
            QNULL!(640),
            QNULL!(768),
            QNULL!(896),
            QNULL!(1024),
            QNULL!(1280),
            QNULL!(1536),
            QNULL!(1792),
            QNULL!(2048),
            QNULL!(2560),
            QNULL!(3072),
            QNULL!(3584),
            QNULL!(4096),
            QNULL!(5120),
            QNULL!(6144),
            QNULL!(7168),
            QNULL!(8192),
            QNULL!(10240),
            QNULL!(12288),
            QNULL!(14336),
            QNULL!(16384),
            QNULL!(20480),
            QNULL!(24576),
            QNULL!(28672),
            QNULL!(32768),
            QNULL!(40960),
            QNULL!(49152),
            QNULL!(57344),
            QNULL!(65536),
            QNULL!(81920),
            QNULL!(98304),
            QNULL!(114688),
            QNULL!(MI_LARGE_WSIZE_MAX + 1), /*131072, Huge queue */
            QNULL!(MI_LARGE_WSIZE_MAX + 2), /* Full queue */
        ]
    };
}

// --------------------------------------------------------
// Statically allocate an empty heap as the initial
// thread local value for the default heap,
// and statically allocate the backing heap for the main
// thread so it can function without doing any allocation
// itself (as accessing a thread local for the first time
// may lead to allocation itself on some platforms)
// --------------------------------------------------------

pub(crate) static MI_HEAP_EMPTY: MiHeap = MiHeap {
    tld: null_mut(),
    pages_free_direct: MI_SMALL_PAGES_EMPTY!(),
    pages: MI_PAGE_QUEUES_EMPTY!(),
    thread_delayed_free: AtomicPtr::new(null_mut()),
    thread_id: 0,
    page_count: 0,
    no_reclaim: false,
};

thread_local! {
    pub(crate) static MI_HEAP_DEFAULT: Cell<*mut MiHeap> = Cell::new(&MI_HEAP_EMPTY as *const MiHeap as *mut MiHeap);
}

pub(crate) static TLD_MAIN: MiTld = MiTld {
    heartbeat: AtomicU64::new(0),
    heap_backing: std::ptr::addr_of!(MI_HEAP_MAIN) as *mut _,
    segments: MiSegmentsTld {
        small_free: MiSegmentQueue {
            first: null_mut(),
            last: null_mut(),
        },
        current_size: 0,
        peak_size: 0,
        cache_count: 0,
        cache_size: 0,
        cache: MiSegmentQueue {
            first: null_mut(),
            last: null_mut(),
        },
        // stats: &TLD_MAIN.stats as *const MiStats as *mut MiStats,
    },
    os: MiOsTld {
        mmap_next_probable: 0,
        mmap_previous: null_mut(),
        pool: null_mut(),
        pool_available: 0,
        // stats: &TLD_MAIN.stats as *const MiStats as *mut MiStats,
    },
    // stats: MiStats::default(),
};

pub(crate) static MI_HEAP_MAIN: MiHeap = MiHeap {
    tld: &TLD_MAIN as *const MiTld as *mut MiTld,
    pages_free_direct: MI_SMALL_PAGES_EMPTY!(),
    pages: MI_PAGE_QUEUES_EMPTY!(),
    thread_delayed_free: AtomicPtr::new(null_mut()),
    thread_id: 0,
    page_count: 0,
    no_reclaim: false, // can reclaim
};

pub(crate) static MI_PROCESS_IS_INITIALIZED: Once = Once::new();

// pub(crate) static mut MI_STATS_MAIN: MiStats = 0;

/* -----------------------------------------------------------
  Initialization of random numbers
----------------------------------------------------------- */

pub(crate) fn mi_random_shuffle(mut x: usize) -> usize {
    if MI_INTPTR_SIZE == 8 {
        // by Sebastiano Vigna, see: <http://xoshiro.di.unimi.it/splitmix64.c>
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
    } else if MI_INTPTR_SIZE == 4 {
        // by Chris Wellons, see: <https://nullprogram.com/blog/2018/07/31/>
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846ca68b);
        x ^= x >> 16;
    }
    return x;
}

pub(crate) fn mi_random_init(seed: usize) -> usize {
    // Hopefully, ASLR makes our function address random
    let mut x = mi_random_init as usize;
    x ^= seed;

    // xor with high res time
    let start = SystemTime::now();
    if let Ok(duration) = start.duration_since(UNIX_EPOCH) {
        x ^= duration.as_secs() as usize;
        x ^= duration.subsec_nanos() as usize;
    }

    // Do a few randomization steps
    let max = ((x ^ (x >> 17)) & 0x0F) + 1;
    for _ in 0..max {
        x = mi_random_shuffle(x);
    }

    x
}

/* -----------------------------------------------------------
  Initialization and freeing of the thread local heaps
----------------------------------------------------------- */
struct MiThreadData {
    pub(crate) heap: MiHeap,
    pub(crate) tld: MiTld,
}

// Initialize the thread local default heap, called from `mi_thread_init`
unsafe fn mi_heap_init() -> bool {
    if mi_heap_is_initialized(mi_get_default_heap()) {
        return true;
    }
    if mi_is_main_thread() {
        // the main heap is statically allocated
        mi_heap_set_default(addr_of!(MI_HEAP_MAIN) as *mut _);
    } else {
        // use `_mi_os_alloc` to allocate directly from the OS
        let td: *mut MiThreadData = {
            let td = mi_os_alloc(std::mem::size_of::<MiThreadData>());
            // Todo: more efficient allocation?
            if td.is_null() {
                eprintln!("failed to allocate thread local heap memory");
                return false;
            }
            td as *mut MiThreadData
        };
        let heap = addr_of_mut!((*td).heap);
        let tld = addr_of_mut!((*td).tld);

        std::ptr::copy_nonoverlapping(&MI_HEAP_EMPTY, heap, 1);
        (*heap).thread_id = mi_thread_id();
        (*heap).tld = tld;

        std::ptr::write_bytes(tld, 0, 1);
        (*tld).heap_backing = heap;

        mi_heap_set_default(heap);
    }
    false
}

// Free the thread local default heap (called from `mi_thread_done`)
pub(crate) fn mi_heap_done() -> bool {
    unsafe {
        let heap = mi_get_default_heap();
        if !mi_heap_is_initialized(heap) {
            return true;
        }

        // reset default heap
        MI_HEAP_DEFAULT.with(|h| {
            h.set(if mi_is_main_thread() {
                addr_of!(MI_HEAP_MAIN) as *mut _
            } else {
                addr_of!(MI_HEAP_EMPTY) as *mut _
            });
        });

        // todo: delete all non-backing heaps?

        // switch to backing heap and free it
        let heap = (*(*heap).tld).heap_backing;
        if !mi_heap_is_initialized(heap) {
            return false;
        }

        // free if not the main thread (or in debug mode)
        if heap != addr_of!(MI_HEAP_MAIN) as *mut _ {
            if (*heap).page_count > 0 {
                mi_heap_collect_abandon(heap);
            }
            mi_os_free(heap as *mut u8, std::mem::size_of::<MiThreadData>());
        }
    }
    false
}

// --------------------------------------------------------
// Try to run `mi_thread_done()` automatically so any memory
// owned by the thread but not yet released can be abandoned
// and re-owned by another thread.
//
// 1. windows dynamic library:
//     call from DllMain on DLL_THREAD_DETACH
// 2. windows static library:
//     use `FlsAlloc` to call a destructor when the thread is done
// 3. unix, pthreads:
//     use a pthread key to call a destructor when a pthread is done
//
// In the last two cases we also need to call `mi_process_init`
// to set up the thread local keys.
// --------------------------------------------------------

static mut MI_PTHREAD_KEY: pthread_key_t = 0;

pub(crate) extern "C" fn mi_pthread_done(value: *mut std::ffi::c_void) {
    if !value.is_null() {
        mi_thread_done();
    }
}

// Set up handlers so `mi_thread_done` is called automatically
pub(crate) fn mi_process_setup_auto_thread_done() {
    static mut TLS_INITIALIZED: bool = false; // fine if it races
    unsafe {
        if TLS_INITIALIZED {
            return;
        }
        TLS_INITIALIZED = true;
    }

    unsafe {
        pthread_key_create(
            &raw const MI_PTHREAD_KEY as *const _ as *mut _,
            Some(mi_pthread_done),
        )
    };
}

pub(crate) fn mi_is_main_thread() -> bool {
    MI_HEAP_MAIN.thread_id == 0 || MI_HEAP_MAIN.thread_id == mi_thread_id()
}

// This is called from the `mi_malloc_generic`
pub(crate) fn mi_thread_init() {
    // ensure our process has started already
    mi_process_init();

    // initialize the thread local default heap
    if unsafe { mi_heap_init() } {
        return; // returns true if already initialized
    }

    // don't further initialize for the main thread
    if mi_is_main_thread() {
        return;
    }

    // set hooks so our mi_thread_done() will be called

    unsafe {
        pthread_setspecific(MI_PTHREAD_KEY, (mi_thread_id() | 1) as *mut _); // set to a dummy value so that `mi_pthread_done` is called
    }
}

pub(crate) fn mi_thread_done() {
    // abandon the thread local heap
    if mi_heap_done() {
        return; // returns true if already ran
    }
}

// --------------------------------------------------------
// Run functions on process init/done, and thread init/done
// --------------------------------------------------------
#[ctor]
fn mi_process_init() {
    // ensure we are called once
    MI_PROCESS_IS_INITIALIZED.call_once(|| {
        unsafe {
            let main_heap_mut = &MI_HEAP_MAIN as *const _ as *mut MiHeap;
            (*main_heap_mut).thread_id = mi_thread_id();
        }

        // Register process cleanup at exit
        // atexit(mi_process_done);

        mi_process_setup_auto_thread_done();
    });
}

#[dtor]
unsafe fn mi_process_done() {
    // only shutdown if we were initialized
    if !MI_PROCESS_IS_INITIALIZED.is_completed() {
        return;
    }

    // ensure we are called once
    static mut PROCESS_DONE: bool = false;
    if PROCESS_DONE {
        return;
    }
    PROCESS_DONE = true;
}
