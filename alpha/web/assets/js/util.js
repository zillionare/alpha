// this provides xpath selector functions
function xpath(xp) {
    const snapshot = document.evaluate(
        xp, document, null, 
        XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null
    );
    return [Array(snapshot.snapshotLength)]
        .map((_, i) => snapshot.snapshotItem(i))
    ;
}

function wave_emit(event_source, event_name) {
    return function(e){
        console.info("wave_emit", event_source, event_name, e.data);
        wave.emit(event_source, event_name, e.data);
    }
}

function bind_event(css_selector, event, callback) {
    el = document.querySelectorAll(css_selector);
    console.info("bind_event found el", el, css_selector)
    if (el.length > 0) {
        el.forEach(element => {
            element.addEventListener(event, callback);
        })
    }else{
        console.warn("bind_event failed: ", css_selector, event);
    }
}