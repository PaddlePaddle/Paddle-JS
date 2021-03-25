/**
 * @file matmul
 */

function mainFunc(
    { origin },
    {
        transpose_X = false,
        transpose_Y = false
    }
) {
    const sharedDim = transpose_X ? origin.height_shape : origin.width_shape;

    return `
    void main() {
        float res = 0.0;
        // 获取output的坐标
        ivec4 out_pos = getOutputTensorPos();
        ivec4 origin_pos = out_pos;
        if (${transpose_X}) {
            origin_pos[3] = origin_pos[2];
        }
        ivec4 counter_pos = out_pos;
        if (${transpose_Y}) {
            counter_pos[2] = counter_pos[3];
        }

        for (int j = 0; j < ${sharedDim}; j++) {
            if (${transpose_X}) {
                origin_pos[2] = j;
            }
            else {
                origin_pos[3] = j;
            }
            if (${transpose_Y}) {
                counter_pos[3] = j;
            }
            else {
                counter_pos[2] = j;
            }
            float o = getValueFromTensorPos_origin(origin_pos[0], origin_pos[1], origin_pos[2], origin_pos[3]);
            float c = getValueFromTensorPos_counter(counter_pos[0], counter_pos[1], counter_pos[2], counter_pos[3]);
            
            res += c * o;
        }
        setOutput(res);
    }
    `;
}
export default {
    mainFunc,
    params: [
        'transpose_X',
        'transpose_Y'
    ],
    textureFuncConf: {
        counter: ['getValueFromTensorPos'],
        origin: ['getValueFromTensorPos']
    }
};
