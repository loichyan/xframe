use crate::{utils::UnwrapThrowValExt, DOCUMENT};
use js_sys::{Function, Object, Reflect};
use std::{borrow::Cow, cell::RefCell, collections::HashMap};
use wasm_bindgen::{prelude::*, JsCast, JsValue};
use web_sys::{Event, EventTarget};
use xframe_core::EventHandler;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(extends = Object)]
    #[derive(Clone)]
    type Descriptor;

    #[wasm_bindgen(method, setter)]
    fn set_configurable(this: &Descriptor, val: bool);

    #[wasm_bindgen(method, setter, js_name = value)]
    fn set_target_value(this: &Descriptor, val: &JsValue);

    #[wasm_bindgen(method, setter, js_name = get)]
    fn set_current_target_get(this: &Descriptor, val: &Function);

    #[wasm_bindgen(method, setter, js_name = value)]
    fn set_stop_propagation_value(this: &Descriptor, val: &Function);

    #[wasm_bindgen(extends = Event)]
    #[derive(Clone, PartialEq, Eq)]
    type EventX;

    #[wasm_bindgen(method, getter, js_name = "$$$CANCEL_BUBBLE")]
    fn cancel_bubble2(this: &EventX) -> JsValue;

    #[wasm_bindgen(method, setter, js_name = "$$$CANCEL_BUBBLE")]
    fn set_cancel_bubble2(this: &EventX, val: bool);

    #[wasm_bindgen(extends = EventTarget)]
    #[derive(Clone, PartialEq, Eq)]
    type EventTargetX;

    #[wasm_bindgen(method, getter, js_name = "$$$EVENT_HANDLERS")]
    fn event_handlers(this: &EventTargetX) -> Option<Object>;

    #[wasm_bindgen(method, setter, js_name = "$$$EVENT_HANDLERS")]
    fn set_event_handlers(this: &EventTargetX, val: &Object);

    #[wasm_bindgen(method, getter)]
    fn disabled(this: &EventTargetX) -> JsValue;

    #[wasm_bindgen(method, getter, js_name = parentNode)]
    fn parent(this: &EventTargetX) -> Option<EventTargetX>;

    #[wasm_bindgen(method, getter)]
    fn host(this: &EventTargetX) -> Option<EventTargetX>;

    #[wasm_bindgen(extends = Object)]
    #[derive(Clone)]
    type CurrentTarget;

    #[wasm_bindgen(method, getter)]
    fn inner(this: &CurrentTarget) -> Option<EventTargetX>;

    #[wasm_bindgen(method, setter)]
    fn set_inner(this: &CurrentTarget, inner: &EventTargetX);
}

type DelegatedEvents = HashMap<&'static str, bool>;

thread_local! {
    static GLOBAL_EVENTS: RefCell<DelegatedEvents> = RefCell::new(delegated_events());
}

fn delegated_events() -> DelegatedEvents {
    [
        "beforeinput",
        "click",
        "contextmenu",
        "dblclick",
        "focusin",
        "focusout",
        "input",
        "keydown",
        "keyup",
        "mousedown",
        "mousemove",
        "mouseout",
        "mouseover",
        "mouseup",
        "pointerdown",
        "pointermove",
        "pointerout",
        "pointerover",
        "pointerup",
        "touchend",
        "touchmove",
        "touchstart",
    ]
    .into_iter()
    .map(|key| (key, false))
    .collect()
}

fn reverse_shadow_dom_retargetting(ev: &EventX, target: &EventTargetX) {
    thread_local! {
        static K_TARGET: JsValue = JsValue::from_str("target");
    }

    let descriptor = Object::new().unchecked_into::<Descriptor>();
    descriptor.set_configurable(true);
    descriptor.set_target_value(target);
    K_TARGET.with(|key| Object::define_property(ev, key, &descriptor));
}

fn simulate_current_target(ev: &EventX, current: &CurrentTarget) {
    thread_local! {
        static K_CURRENT_TARGET: JsValue = JsValue::from_str("currentTarget");
    }

    let current = current.clone();
    let descriptor = Object::new().unchecked_into::<Descriptor>();
    descriptor.set_configurable(true);
    descriptor.set_current_target_get(
        &Closure::<dyn FnMut() -> Option<EventTargetX>>::new(move || current.inner())
            .into_js_value()
            .unchecked_into(),
    );
    K_CURRENT_TARGET.with(|key| Object::define_property(ev, key, &descriptor));
}

fn hook_stop_propagation(ev: &EventX) {
    thread_local! {
        static K_STOP_PROPAGATION: JsValue = JsValue::from_str("stopPropagation");
    }

    let descriptor = Object::new().unchecked_into::<Descriptor>();
    descriptor.set_configurable(true);
    descriptor.set_stop_propagation_value({
        let ev = ev.clone();
        &Closure::<dyn FnMut()>::new(move || {
            ev.set_cancel_bubble2(true);
            ev.stop_propagation();
        })
        .into_js_value()
        .unchecked_into()
    });
    K_STOP_PROPAGATION.with(|key| Object::define_property(ev, key, &descriptor));
}

fn event_handler(name: JsValue) -> Function {
    let handler = move |ev: Event| {
        let ev: EventX = ev.unchecked_into();
        let target: EventTargetX = ev.target().unwrap().unchecked_into();
        let node: EventTargetX = {
            let node = ev.composed_path().get(0);
            if node.is_undefined() {
                target.clone()
            } else {
                node.unchecked_into()
            }
        };
        if target != node {
            reverse_shadow_dom_retargetting(&ev, &node);
        }
        let shared_node = Object::new().unchecked_into::<CurrentTarget>();
        simulate_current_target(&ev, &shared_node);
        // Simulate event bubbling.
        hook_stop_propagation(&ev);
        let mut current = node;
        loop {
            shared_node.set_inner(&current);
            if current.disabled().is_falsy() {
                if let Some(handlers) = current.event_handlers() {
                    let handler = Reflect::get(&handlers, &name).unwrap_throw_val();
                    if !handler.is_undefined() {
                        handler
                            .unchecked_into::<Function>()
                            .call1(&current, &ev)
                            .unwrap_throw_val();
                    }
                }
            }
            if ev.cancel_bubble2().is_truthy() {
                break;
            }
            if let Some(next) = current.parent().or_else(|| current.host()) {
                current = next;
            } else {
                DOCUMENT.with(|doc| shared_node.set_inner(doc.unchecked_ref()));
                break;
            }
        }
    };

    Closure::<dyn FnMut(Event)>::new(handler)
        .into_js_value()
        .unchecked_into()
}

pub fn add_event_listener(
    node: &web_sys::Node,
    event: Cow<'static, str>,
    handler: EventHandler<Event>,
) {
    let EventHandler { handler, options } = handler;
    let handler = Closure::wrap(handler)
        .into_js_value()
        .unchecked_into::<Function>();
    GLOBAL_EVENTS.with(|events| {
        if !options.capture {
            let mut events = events.borrow_mut();
            if let Some(registered) = events.get_mut(&*event) {
                // Inject event handler.
                let name = JsValue::from_str(&event);
                let target = node.unchecked_ref::<EventTargetX>();
                let handlers = target.event_handlers().unwrap_or_else(|| {
                    let new = Object::new().unchecked_into();
                    target.set_event_handlers(&new);
                    new
                });
                Reflect::set(&handlers, &name, &handler).unwrap_throw_val();

                // Register global event handler.
                if !*registered {
                    *registered = true;
                    DOCUMENT.with(|doc| {
                        doc.add_event_listener_with_callback(&event, &event_handler(name))
                            .unwrap_throw_val();
                    });
                }

                return;
            }
        }
        // Don't delegate the event.
        node.add_event_listener_with_callback_and_bool(&event, &handler, options.capture)
            .unwrap_throw_val();
    });
}
