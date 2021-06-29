/**
 * @file op 相关辅助函数
 * @author yueshuangyan
 */

import { Tensor } from '../types';

interface opInfo {
    [key: string]: any
}

const inputParams = [
    'length_shape',
    'length_unformatted_shape',
    'width_shape',
    'height_shape',
    'width_texture',
    'height_texture',
    'offset_x',
    'offset_y',
    'limit',
    'channel',
    'total_shape',
    'numbers_shape'
];

const outParams = [
    'total_shape',
    'width_shape',
    'height_shape',
    'width_texture',
    'height_texture',
    'channel',
    'limit'
];

const baseParams = {
    float: [
        'multi_value',
        'bias_value'
    ],
    bool: [
        'fuse_relu'
    ]
};

function getTensorParams(
    inputTensors: Tensor[],
    ownParams: [],
    fShaderParams: object,
    runtime: number,
    key: string
): opInfo {
    const tensorsParams = {};
    const opParams = {};
    const tensorNames = [] as string[];

    // inputParams
    for (const tensor of inputTensors) {
        const name = tensor.name;
        // 提取inputParams
        const inputVars = {};
        for (const param of inputParams) {
            if (typeof (tensor[param]) !== 'undefined') {
                inputVars[param] = tensor[param];
            }
        }

        tensorsParams[name] = inputVars;
        tensorNames.push(name);
    }


    // ownParams
    if (ownParams) {
        for (const param of ownParams) {
            if (typeof (fShaderParams[param]) !== 'undefined') {
                opParams[param] = fShaderParams[param];
            }
        }
    }

    // baseParams
    for (const type of Object.keys(baseParams)) {
        const params = baseParams[type];
        for (const param of params) {
            if (typeof (fShaderParams[param]) !== 'undefined') {
                opParams[param] = `${type}(${fShaderParams[param]})`;
            }
        }
    }

    // outputParams
    const outputVars = {};
    for (const param of outParams) {
        if (typeof (fShaderParams[param + '_out']) !== 'undefined') {
            outputVars[param] = fShaderParams[param + '_out'];
        }
    }
    tensorsParams['out'] = outputVars;

    // 将active_function放在opParams中
    if (fShaderParams['active_function']) {
        opParams['active_function'] = fShaderParams['active_function'];
    }

    opParams['runtime'] = runtime;
    opParams['key'] = key;

    return { textureParams: tensorsParams, opParams, active_function: fShaderParams['active_function'] };
}


function getExactOpName(name, isPacked) {
    if (name.indexOf('conv2d-elementwise_add') > -1) {
        return 'conv2d_elementwise_add';
    }
    return isPacked ? `${name}_packing` : name;
}

export {
    getExactOpName,
    getTensorParams
};
