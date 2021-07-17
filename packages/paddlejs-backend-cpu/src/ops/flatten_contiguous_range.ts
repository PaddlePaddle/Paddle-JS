import { Tensor } from './Tensor';
import { getInt } from './utils';

/* eslint-disable max-statements */
function main(tensorMap: Map<string, Tensor>, attrs: Attrs, runtime: i32): f32[] {
    // v1 paddle.fluid parameter
    let axis = attrs.axis;

    // v2 paddle.nn parameter
    let start_axis = attrs.start_axis;
    let stop_axis = attrs.stop_axis;

    // generic parameters
    const origin = tensorMap.get('origin') as Tensor;
    const out = tensorMap.get('out_' + runtime) as Tensor;    

    const outShape: i32[] = (out as Tensor).shape;
    const startIndex = out.runtime * length;
    const originReducedShape: i32[] = (origin as Tensor).shapeReduced;
    const outReducedShape: i32[] = (out as Tensor).shapeReduced;

    const originData: f32[] = (origin as Tensor).data;

    const outN = outShape[0];
    const outC = outShape[1];
    const outH = outShape[2];
    const outW = outShape[3];

    const originS0 = originReducedShape[0];
    const originS1 = originReducedShape[1];
    const originS2 = originReducedShape[2];

    const outS0 = outReducedShape[0];
    const outS1 = outReducedShape[1];
    const outS2 = outReducedShape[2];

    const result = new Array<f32>(out.total);

    for (let n = 0; n < outN; n++) {
        for (let c = 0; c < outC; c++) {
            for (let h = 0; h < outH; h++) {
                for (let w = 0; w < outW; w++) {

                    result[n * originS0 + c * originS1 + h * originS2 + w]
                        = originData[n * originS0 + c * originS1 + h * originS2 + w];
                        
                }
            }
        }
    }
    return result;
}
/* eslint-enable max-statements */

class Attrs {
    axis: i32 = 1;

    start_axis : i32 = 0;
    stop_axis : i32 = -1;

    constructor(data: Obj) {
        this.start_axis = getInt('start_axis', data);
        this.stop_axis = getInt('stop_axis', data);
        this.axis = getInt('axis', data);
    }
}

const behaviors = [];

export {
    main,
    Attrs,
    behaviors
};