// this provides xpath selector functions
const xpath = xp => {
    const snapshot = document.evaluate(
        xp, document, null, 
        XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null
    );
    return [Array(snapshot.snapshotLength)]
        .map((_, i) => snapshot.snapshotItem(i))
    ;
};

function wave_emit(event_source, event_name, event_data) {
    wave.emit(event_source, event_name, event_data);
}

function bind_event(xpath_selector, event) {
    el = xpath(xpath_selector)
    if (el.length > 0) {
        el.array.forEach(element => {
            element.addEventListener(event, callback);
        })
    }
}
