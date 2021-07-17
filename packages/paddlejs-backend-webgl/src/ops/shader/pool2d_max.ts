/**
 * @file pool2d_max
 */

function recoverShape({ total_shape, channel, height_shape, width_shape }) {
    return [total_shape / channel / height_shape / width_shape, channel, height_shape, width_shape];
}

function mainFunc(
    { origin },
    { strides = [], paddings = [], ksize, global_pooling, key }
) {
    const [stride_v = 1, stride_h = 1] = strides;
    const [padTop = 0, padLeft = 0] = paddings;
    const [ksize_x, ksize_y] = ksize;
    const originShape = recoverShape(origin);
    let computedIndex = '';
    let outputCode = 'setOutput(float(res));';
    if (key === 'Mask' && global_pooling === true) {
        computedIndex = `
            if (curr > res) {
                index = ${originShape[2] * originShape[3]} * out_pos[1] + ${originShape[3]} * oy + ox;
            }
        `;
        outputCode = 'setOutput(float(index));';
    }

    return `
    // start函数
    void main(void) {
        float res = -1. / 0.;
        int index = 0;
        // 获取output的坐标
        ivec4 out_pos = getOutputTensorPos();
        int b = out_pos[0];
        int c = out_pos[1];
        int y = out_pos[2];
        int x = out_pos[3];
        // X、Y方向的移动步长
        int oy_base = out_pos[2] * ${stride_v} - ${padTop};
        int ox_base = out_pos[3] * ${stride_h} - ${padLeft};
        for (int fy = 0; fy < ${ksize_y}; fy++) {
            int oy = oy_base + fy;
            if (oy >= ${origin.height_shape}) {
                break;
            }
            if (oy < 0) {
                continue;
            }
            for (int fx = 0; fx < ${ksize_x}; fx++) {
                int ox = ox_base + fx;
                if (ox >= ${origin.width_shape}) {
                    break;
                }
                if (ox < 0) {
                    continue;
                }
                // origin数据
                float curr = getValueFromTensorPos_origin(out_pos[0], out_pos[1], oy, ox);
                ${computedIndex}
                res = max(res, curr);
            }
        }
        ${outputCode}
    }
    `;
}
export default {
    mainFunc,
    params: [
        'strides',
        'paddings',
        'ksize',
        'global_pooling',
        'runtime',
        'key'
    ],
    textureFuncConf: {
        origin: ['getValueFromTensorPos']
    },
    behaviors: [
        'isMax',
        'setPacked',
        'setAdaptive',
        'isGlobalPooling'
    ]
};
