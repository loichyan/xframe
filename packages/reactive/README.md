# xframe-reactive

The reactive system of `xFrame`, provides easy-to-use APIs to build fine-grained
reactive applications.

## ✍️ Example

```rs
use xframe_reactive::*;

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
`xframe-reactive`.

- [sycamore-rs/sycamore](https://github.com/sycamore-rs/sycamore): A library for
  creating reactive web apps in Rust and WebAssembly.
- [gbj/leptos](https://github.com/gbj/leptos): A full-stack, isomorphic Rust web
  framework leveraging fine-grained reactivity to build declarative user
  interfaces.
