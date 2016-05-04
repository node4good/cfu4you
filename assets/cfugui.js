'use strict'
function set_canvas() {
    var e = new Image();
    e.addEventListener("load", function() {
        var imgInstance = new fabric.Image(e);
        var c = document.getElementById('c');
        c.width = imgInstance.width;
        c.height = imgInstance.height;

        var canvas = new fabric.Canvas(c);
        canvas.setBackgroundImage(imgInstance);
        var dp = d.map(x => x.contour.map(a => a[0]))
            .map(c => 'M' + c.map(p => p.join(',')).join('L') + 'z');
        var paths = dp.map(c => new fabric.Path(c).set({ fill: 'red', stroke: 'green', opacity: 0.5 }));
        console.log('created')
        canvas.add(...paths)
        console.log('added')
        document.onkeyup = (e) => {
            if(e.keyCode == 46) {
                canvas.remove(canvas.getActiveObject());
            }
        }
    }, false);
    e.src = img_src;
}
set_canvas();
