//! # Broom
//!
//! An ergonomic tracing garbage collector that supports mark 'n sweep garbage collection.
//!
//! ## Example
//!
//! ```
//! use broom::prelude::*;
//!
//! // The type you want the heap to contain
//! pub enum Object {
//!     Num(f64),
//!     List(Vec<Handle<Self>>),
//! }
//!
//! // Tell the garbage collector how to explore a graph of this object
//! impl Trace<Self> for Object {
//!     fn trace(&self, tracer: &mut Tracer<Self>) {
//!         match self {
//!             Object::Num(_) => {},
//!             Object::List(objects) => objects.trace(tracer),
//!         }
//!     }
//! }
//!
//! // Create a new heap
//! let mut heap = Heap::default();
//!
//! // Temporary objects are cheaper than rooted objects, but don't survive heap cleans
//! let a = heap.insert_temp(Object::Num(42.0));
//! let b = heap.insert_temp(Object::Num(1337.0));
//!
//! // Turn the numbers into a rooted list
//! let c = heap.insert(Object::List(vec![a, b]));
//!
//! // Change one of the numbers - this is safe, even if the object is self-referential!
//! *heap.get_mut(a).unwrap() = Object::Num(256.0);
//!
//! // Create another number object
//! let d = heap.insert_temp(Object::Num(0.0));
//!
//! // Clean up unused heap objects
//! heap.clean();
//!
//! // a, b and c are all kept alive because c is rooted and a and b are its children
//! assert!(heap.contains(a));
//! assert!(heap.contains(b));
//! assert!(heap.contains(c));
//!
//! // Because `d` was temporary and unused, it did not survive the heap clean
//! assert!(!heap.contains(d));
//!
//! ```

pub mod trace;

use crate::trace::*;
use hashbrown::{HashMap, HashSet};
//use std::collections::{HashMap, HashSet};
use std::{
    hash::{Hash, Hasher},
    //rc::Rc,
    ops::{Deref, DerefMut},
    //cell::{Ref, RefCell, RefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Common items that you'll probably need often.
pub mod prelude {
    pub use super::{
        trace::{Trace, Tracer},
        Handle, Heap, Obj, ObjMut, Rooted,
    };
}

/// A heap for storing objects.
///
/// [`Heap`] is the centre of `broom`'s universe. It's the singleton through with manipulation of
/// objects occurs. It can be used to create, access, mutate and garbage-collect objects.
///
/// Note that heaps, and the objects associated with them, are *not* compatible: this means that
/// you may not create trace routes (see [`Trace`]) that cross the boundary between different heaps.
pub struct Heap<T> {
    last_sweep: usize,
    object_sweeps: HashMap<usize, usize>,
    objects: RwLock<Vec<Option<Arc<RwLock<T>>>>>,
    freed: Vec<usize>,
    rooted: HashMap<HandleRef<T>, Arc<()>>,
    refs: HashMap<Handle<T>, HandleRef<T>>,
}

impl<T> Default for Heap<T> {
    fn default() -> Self {
        Self {
            last_sweep: 0,
            object_sweeps: HashMap::default(),
            objects: RwLock::new(Vec::default()),
            freed: Vec::default(),
            rooted: HashMap::default(),
            refs: HashMap::default(),
        }
    }
}

impl<T: Trace<T>> Heap<T> {
    /// Create an empty heap.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn objects(&self) -> usize {
        self.objects.read().unwrap().len()
    }
    pub fn free_objects(&self) -> usize {
        self.freed.len()
    }
    pub fn used_objects(&self) -> usize {
        self.objects() - self.free_objects()
    }

    fn insert_int(&mut self, object: T) -> (Handle<T>, HandleRef<T>) {
        let idx = if let Some(idx) = self.freed.pop() {
            if let Ok(mut objects) = self.objects.write() {
            objects.push(Some(Arc::new(RwLock::new(object))));
            objects.swap_remove(idx);
            idx
            } else {
                panic!("Poisoned objects list!");
            }
        } else {
            if let Ok(mut objects) = self.objects.write() {
            objects.push(Some(Arc::new(RwLock::new(object))));
            objects.len() - 1
            } else {
                panic!("Poisoned objects list!");
            }
        };
        let handle = Handle {
            idx,
            _marker: std::marker::PhantomData,
        };
        let href = HandleRef { handle, idx };
        self.refs.insert(handle, href);
        (handle, href)
    }

    /// Adds a new object to this heap that will be cleared upon the next garbage collection, if
    /// not attached to the object tree.
    pub fn insert_temp(&mut self, object: T) -> Handle<T> {
        let (handle, _) = self.insert_int(object);
        handle
    }

    /// Adds a new object to this heap that will not be cleared by garbage collection until all
    /// rooted handles have been dropped.
    pub fn insert(&mut self, object: T) -> Rooted<T> {
        let (handle, href) = self.insert_int(object);

        let rc = Arc::new(());
        self.rooted.insert(href, rc.clone());

        Rooted { rc, handle }
    }

    /// Upgrade a handle (that will be cleared by the garbage collector) into a rooted handle (that
    /// will not).
    pub fn make_rooted(&mut self, handle: impl AsRef<Handle<T>>) -> Rooted<T> {
        let handle = handle.as_ref();
        debug_assert!(self.contains(handle));

        if let Some(href) = self.refs.get(handle) {
        Rooted {
            rc: self
                .rooted
                .entry(*href)
                .or_insert_with(|| Arc::new(()))
                .clone(),
            handle: *handle,
        }
        } else {
            panic!("Not a valid handle!");
        }
    }

    /// Count the number of heap-allocated objects in this heap
    pub fn len(&self) -> usize {
        self.objects.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.read().unwrap().is_empty()
    }

    /// Return true if the heap contains the specified handle
    pub fn contains(&self, handle: impl AsRef<Handle<T>>) -> bool {
        let handle = handle.as_ref();
        if let Some(href) = self.refs.get(handle) {
            self.objects.read().expect("Poisoned object list!").get(href.idx).expect("Invalid handle!").is_some()
        } else {
            false
        }
    }

    /// Get a reference to a heap object if it exists on this heap.
    pub fn get(&self, handle: impl AsRef<Handle<T>>) -> Option<Obj<T>> {
        let handle = handle.as_ref();
        if let Some(href) = self.refs.get(handle) {
            if let Some(obj) = self.objects.read().unwrap().get(href.idx) {
                if let Some(obj) = obj {
                Some(Obj::new(obj.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get a mutable reference to a heap object
    pub fn get_mut(&self, handle: impl AsRef<Handle<T>>) -> Option<ObjMut<T>> {
        let handle = handle.as_ref();
        if let Some(href) = self.refs.get(handle) {
            if let Some(obj) = self.objects.read().unwrap().get(href.idx) {
                if let Some(obj) = obj {
                Some(ObjMut::new(obj.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /*pub fn as_ptr(&mut self, handle: impl AsRef<Handle<T>>) -> *mut T {
        let handle = handle.as_ref();
        self.objects[handle.idx].as_ptr()
    }*/

    /// Clean orphaned objects from the heap, excluding those that can be reached from the given
    /// handle iterator.
    ///
    /// This function is useful in circumstances in which you wish to keep certain items alive over
    /// a garbage collection without the addition cost of a [`Rooted`] handle. An example of this
    /// might be stack items in a garbage-collected language
    pub fn clean_excluding(&mut self, excluding: impl IntoIterator<Item = Handle<T>>) {
        let new_sweep = self.last_sweep + 1;
        let mut tracer = Tracer {
            new_sweep,
            object_sweeps: &mut self.object_sweeps,
            objects: &self.objects,
            refs: &self.refs,
        };

        // Mark
        let objects = &self.objects;
        self.rooted.retain(|href, rc| {
            if Arc::strong_count(rc) > 1 {
                tracer.mark(href);
                if let Some(obj) = &objects.read().unwrap().get(href.idx) {
                    //obj.borrow().trace(&mut tracer);
                    // XXX FIX ME
                    if let Some(obj) = obj {
                    obj.read().unwrap().trace(&mut tracer);
                    }
                }
                true
            } else {
                false
            }
        });
        excluding
            .into_iter()
            .for_each(|handle| {
                tracer.mark_handle(&handle);
                if let Some(obj) = &objects.read().unwrap().get(handle.idx) {
                    //obj.borrow().trace(&mut tracer);
                    // XXX FIX ME
                    if let Some(obj) = obj {
                        obj.read().unwrap().trace(&mut tracer);
                    }
                }
            });

        // Sweep
        for (i, obj) in self.objects.write().unwrap().iter_mut().enumerate() {
            if !self
                .object_sweeps
                .get(&i)
                .map(|sweep| *sweep == new_sweep)
                .unwrap_or(false)
            {
                self.object_sweeps.remove(&i);
                if obj.is_some() {
                    self.freed.push(i);
                    *obj = None;
                    self.refs.remove(&Handle { idx: i, _marker: std::marker::PhantomData });
                }
            }
        }

        self.last_sweep = new_sweep;
    }

    /// Clean orphaned objects from the heap.
    pub fn clean(&mut self) {
        self.clean_excluding(std::iter::empty());
    }
}

pub struct HandleRef<T> {
    handle: Handle<T>,
    idx: usize,
}

impl<T> Copy for HandleRef<T> {}
impl<T> Clone for HandleRef<T> {
    fn clone(&self) -> HandleRef<T> {
        HandleRef {
            handle: self.handle,
            idx: self.idx,
        }
    }
}

impl<T> PartialEq<Self> for HandleRef<T> {
    fn eq(&self, other: &Self) -> bool {
        //self.gen == other.gen && self.ptr == other.ptr
        self.handle == other.handle
    }
}
impl<T> Eq for HandleRef<T> {}

impl<T> Hash for HandleRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}


/// A handle to a heap object.
///
/// [`Handle`] may be cheaply copied as is necessary to serve your needs. It's even legal for it
/// to outlive the object it refers to, provided it is no longer used to access it afterwards.
#[derive(Debug)]
pub struct Handle<T> {
    idx: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Trace<T>> Handle<T> {}

impl<T> Copy for Handle<T> {}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Handle<T> {
        Handle {
            idx: self.idx,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> PartialEq<Self> for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        //self.gen == other.gen && self.ptr == other.ptr
        self.idx == other.idx
    }
}
impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        //self.gen.hash(state);
        //self.ptr.hash(state);
        self.idx.hash(state);
    }
}

impl<T> AsRef<Handle<T>> for Handle<T> {
    fn as_ref(&self) -> &Handle<T> {
        self
    }
}

impl<T> From<Rooted<T>> for Handle<T> {
    fn from(rooted: Rooted<T>) -> Self {
        rooted.handle
    }
}

pub struct Obj<'a, T> {
    _obj: Arc<RwLock<T>>,
    read: RwLockReadGuard<'a, T>,
}

impl<'a, T> Obj<'a, T> {
    fn new_lifetime(obj: &Arc<RwLock<T>>) -> &'a Arc<RwLock<T>> {
        unsafe { &*(obj as *const Arc<RwLock<T>>) }
    }

    fn new(obj: Arc<RwLock<T>>) -> Obj<'a, T> {
        let read = Obj::new_lifetime(&obj)
            .read()
            .expect("Heap has poisoned data, done!");
        Obj { _obj: obj, read }
    }

    pub fn data<'b>(&'b self) -> &'b T
    where
        'a: 'b,
    {
        &*self.read
    }
}

impl<'a, T> Deref for Obj<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.read
    }
}

pub struct ObjMut<'a, T> {
    _obj: Arc<RwLock<T>>,
    write: RwLockWriteGuard<'a, T>,
}

impl<'a, T> ObjMut<'a, T> {
    fn new_lifetime(obj: &Arc<RwLock<T>>) -> &'a Arc<RwLock<T>> {
        unsafe { &*(obj as *const Arc<RwLock<T>>) }
    }

    fn new(obj: Arc<RwLock<T>>) -> ObjMut<'a, T> {
        let write = ObjMut::new_lifetime(&obj)
            .try_write()
            .expect("Heap has poisoned data, done!");
        ObjMut { _obj: obj, write }
    }

    pub fn data<'b>(&'b self) -> &'b T
    where
        'a: 'b,
    {
        &*self.write
    }

    pub fn data_mut<'b>(&'b mut self) -> &'b mut T
    where
        'a: 'b,
    {
        &mut *self.write
    }
}

impl<'a, T> Deref for ObjMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.write
    }
}

impl<'a, T> DerefMut for ObjMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.write
    }
}

/// A handle to a heap object that guarantees the object will not be cleaned up by the garbage
/// collector.
///
/// [`Rooted`] may be cheaply copied as is necessary to serve your needs. It's even legal for it
/// to outlive the object it refers to, provided it is no longer used to access it afterwards.
#[derive(Debug)]
pub struct Rooted<T> {
    // TODO: Is an Rc the best we can do? It might be better instead to store the strong count with
    // the object to avoid an extra allocation.
    //rc: Rc<()>,
    rc: Arc<()>,
    handle: Handle<T>,
}

impl<T: Trace<T>> Clone for Rooted<T> {
    fn clone(&self) -> Self {
        Self {
            rc: self.rc.clone(),
            handle: self.handle,
        }
    }
}

impl<T> AsRef<Handle<T>> for Rooted<T> {
    fn as_ref(&self) -> &Handle<T> {
        &self.handle
    }
}

impl<T: Trace<T>> Rooted<T> {
    pub fn into_handle(self) -> Handle<T> {
        self.handle
    }

    pub fn handle(&self) -> Handle<T> {
        self.handle
    }

    pub fn rc(&self) -> Arc<()> {
        self.rc.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    enum Value<'a> {
        Base(&'a AtomicUsize),
        Refs(&'a AtomicUsize, Handle<Value<'a>>, Handle<Value<'a>>),
    }

    impl<'a> Trace<Self> for Value<'a> {
        fn trace(&self, tracer: &mut Tracer<Self>) {
            match self {
                Value::Base(_) => {}
                Value::Refs(_, a, b) => {
                    a.trace(tracer);
                    b.trace(tracer);
                }
            }
        }
    }

    impl<'a> Drop for Value<'a> {
        fn drop(&mut self) {
            match self {
                Value::Base(count) | Value::Refs(count, _, _) => {
                    count.fetch_sub(1, Ordering::Relaxed)
                }
            };
        }
    }

    #[test]
    fn basic() {
        let count: AtomicUsize = AtomicUsize::new(0);

        let new_count = || {
            count.fetch_add(1, Ordering::Relaxed);
            &count
        };

        let mut heap = Heap::default();

        let a = heap.insert(Value::Base(new_count()));

        heap.clean();

        assert_eq!(heap.contains(&a), true);

        let a = a.into_handle();

        heap.clean();

        assert_eq!(heap.contains(&a), false);

        drop(heap);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn ownership() {
        let count: AtomicUsize = AtomicUsize::new(0);

        let new_count = || {
            count.fetch_add(1, Ordering::Relaxed);
            &count
        };

        let mut heap = Heap::default();

        let a = heap.insert(Value::Base(new_count())).handle();
        let b = heap.insert(Value::Base(new_count())).handle();
        let c = heap.insert(Value::Base(new_count())).handle();
        let d = heap.insert(Value::Refs(new_count(), a, c));
        let e = heap.insert(Value::Base(new_count())).handle();

        heap.clean();

        assert_eq!(heap.contains(&a), true);
        assert_eq!(heap.contains(&b), false);
        assert_eq!(heap.contains(&c), true);
        assert_eq!(heap.contains(&d), true);
        assert_eq!(heap.contains(&e), false);

        let a = heap.insert_temp(Value::Base(new_count()));

        heap.clean();
        assert_eq!(heap.contains(&a), false);

        let a = heap.insert_temp(Value::Base(new_count()));
        let a = heap.make_rooted(a);

        heap.clean();

        assert_eq!(heap.contains(&a), true);

        drop(heap);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn recursive() {
        let count: AtomicUsize = AtomicUsize::new(0);

        let new_count = || {
            count.fetch_add(1, Ordering::Relaxed);
            &count
        };

        let mut heap = Heap::default();

        let a = heap.insert(Value::Base(new_count()));
        let b = heap.insert(Value::Base(new_count()));

        *heap.get_mut(&a).unwrap() = Value::Refs(new_count(), a.handle(), b.handle());

        heap.clean();

        assert_eq!(heap.contains(&a), true);
        assert_eq!(heap.contains(&b), true);

        let a = a.into_handle();

        heap.clean();

        assert_eq!(heap.contains(&a), false);
        assert_eq!(heap.contains(&b), true);

        drop(heap);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn temporary() {
        let count: AtomicUsize = AtomicUsize::new(0);

        let new_count = || {
            count.fetch_add(1, Ordering::Relaxed);
            &count
        };

        let mut heap = Heap::default();

        let a = heap.insert_temp(Value::Base(new_count()));

        heap.clean();

        assert_eq!(heap.contains(&a), false);

        let a = heap.insert_temp(Value::Base(new_count()));
        let b = heap.insert(Value::Refs(new_count(), a, a));

        heap.clean();

        assert_eq!(heap.contains(&a), true);
        assert_eq!(heap.contains(&b), true);

        let a = heap.insert_temp(Value::Base(new_count()));

        heap.clean_excluding(Some(a));

        assert_eq!(heap.contains(&a), true);

        drop(heap);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }
}
