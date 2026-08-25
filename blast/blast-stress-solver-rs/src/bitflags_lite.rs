// A small stand-in for the `bitflags` crate.
//
// Deliberately not a dependency: this crate builds with nothing but `cc`, and
// keeping the dependency surface empty matters for the published-crate and
// wasm stories.

macro_rules! bitflags_lite {
    ($(#[$m:meta])* pub struct $name:ident : $ty:ty { $($(#[$fm:meta])* const $f:ident = $v:expr;)* }) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
        pub struct $name($ty);
        impl $name {
            $($(#[$fm])* pub const $f: $name = $name($v);)*
            pub const NONE: $name = $name(0);
            pub const fn bits(self) -> $ty { self.0 }
            pub const fn contains(self, o: $name) -> bool { (self.0 & o.0) == o.0 }
            pub const fn union(self, o: $name) -> $name { $name(self.0 | o.0) }
            pub fn set(&mut self, o: $name, on: bool) {
                if on { self.0 |= o.0 } else { self.0 &= !o.0 }
            }
        }
        impl core::ops::BitOr for $name {
            type Output = $name;
            fn bitor(self, o: $name) -> $name { self.union(o) }
        }
    };
}
