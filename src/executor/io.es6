/**
 * @file io，loader相关输入输出
 * @author wangqun@baidu.com
 */
export default class io {
    constructor() {
        this.fromPixels2DContext = document.createElement('canvas').getContext('2d');
    };

    fromPixels(pixels, numChannels) {
        if (pixels == null) {
            throw new Error(
                'pixels passed to tf.browser.fromPixels() can not be null');
        }
        let vals;
        // tslint:disable-next-line:no-any
        // tslint:disable-next-line:no-any
        if (pixels.getContext != null) {
            // tslint:disable-next-line:no-any
            vals = pixels
                .getContext('2d')
                .getImageData(0, 0, pixels.width, pixels.height)
                .data;
        } else if (pixels instanceof ImageData) {
            vals = pixels.data;
        } else if (
            pixels instanceof HTMLImageElement ||
            pixels instanceof HTMLVideoElement) {
            if (this.fromPixels2DContext == null) {
                throw new Error(
                    'Can\'t read pixels from HTMLImageElement outside ' +
                    'the browser.');
            }
            this.fromPixels2DContext.canvas.width = pixels.width;
            this.fromPixels2DContext.canvas.height = pixels.height;
            this.fromPixels2DContext.drawImage(
                pixels, 0, 0, pixels.width, pixels.height);
            vals = this.fromPixels2DContext
                .getImageData(0, 0, pixels.width, pixels.height)
                .data;
        } else {

        }
        let values;
        if (numChannels === 4) {
            values = new Int32Array(vals);
        } else {
            const numPixels = pixels.width * pixels.height;
            values = new Int32Array(numPixels * numChannels);
            for (let i = 0; i < numPixels; i++) {
                for (let channel = 0; channel < numChannels; ++channel) {
                    values[i * numChannels + channel] = vals[i * 4 + channel];
                }
            }
        }
        console.log(pixels.height, pixels.width, numChannels, values);
        // const outShape: [number, number, number] =
        //     [pixels.height, pixels.width, numChannels];
        return [values, numChannels, 'int32'];
    }

}