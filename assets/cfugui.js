'use strict'
var e = document.getElementById('roi_color.png');
var imgInstance = new fabric.Image(e);
var dp = d.map(x => x.contour.map(a => a[0]))
    .map(c => 'M' + c.map(p => p.join(',')).join('L') + 'z');
var canvas = new fabric.Canvas('c');
canvas.add(imgInstance);
var paths = dp.map(c => new fabric.Path(c).set({ fill: 'red', stroke: 'green', opacity: 0.5 }));
document.getElementById('c').scrollIntoView();
document.onkeyup = (e) => {
    if(e.keyCode == 46) {
        canvas.remove(canvas.getActiveObject());
    }
}
