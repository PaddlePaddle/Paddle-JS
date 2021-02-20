import { Runner } from '@paddlejs/paddlejs-core';
import backend from '@paddlejs/paddlejs-backend-webgpu';


const opName = 'elementwise_add';
const modelDir = '/test/op/data/';
const modelPath = `${modelDir}${opName}.json`;

const OP_INFO_MAP = {
    conv2d: {
        name: 'conv2d_0.tmp_0',
        shape: [
            1,
            4,
            2,
            2
        ]
    },
    elementwise_add: {
        name: 'fc_0.tmp_1',
        shape: [
            3, 2, 2
        ]
    },
    mul: {
        name: 'fc_0.tmp_0',
        shape: [2, 3]
    },
    concat: {
        name: 'concat.tmp_2',
        shape: [2, 4, 2]
    },
    concat_mul: {
        name: 'concat.tmp_3',
        shape: [2, 2, 5]
    },
    split: {
        name: 'split_1.tmp_1',
        shape: [2, 2, 1, 3]
    }
};

async function run() {
    const runner = new Runner({
        modelPath,
        feedShape: {
            fw: 3,
            fh: 3
        },
        fileCount: 0,
        needPreheat: false
    });
    await runner.init();
    const executeOP = runner.weightMap[0];
    runner.executeOp(executeOP);
    console.log(await backend.read(OP_INFO_MAP[opName]));
}

run();
