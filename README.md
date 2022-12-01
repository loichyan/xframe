# 🍃 xFrame

A reactive system to build fine-grained reactive applications.

## ✍️ Example

```rust
use xframe::*;

create_root(|cx| {
    let state = cx.create_signal(1);

    let double = cx.create_memo(move || *state.get() * 2);
    assert_eq!(*double.get(), 2);

    state.set(2);
    assert_eq!(*double.get(), 4);

    state.set(3);
    assert_eq!(*double.get(), 6);
});
```

## 💭 Insipired by

Please check out these awesome works that helped a lot in the creation of
`xframe`:

- [sycamore-rs/sycamore](https://github.com/sycamore-rs/sycamore): A library for
  creating reactive web apps in Rust and WebAssembly.
- [gbj/leptos](https://github.com/gbj/leptos): A full-stack, isomorphic Rust web
  framework leveraging fine-grained reactivity to build declarative user
  interfaces.

## ⚖️ License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.
