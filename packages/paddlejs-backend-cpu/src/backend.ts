/**
 * @file webgl backend
 * @author yueshuangyan
 */

import { PaddlejsBackend, env } from '@paddlejs/paddlejs-core';
import Program from './Program';
import { OpData } from './types';
import { Tensor } from './ops/Tensor';


export default class CpuBackend extends PaddlejsBackend {
    program?: Program;
    dataMap?: Map<string, number[]>;
    noLayout: boolean;
    constructor() {
        super();
        this.dataMap = new Map();
        this.noLayout = true;
    }

    async init() {
    }

    createProgram({ name, outTensor }) {
        const outName = outTensor.tensorId;
        if (!outName) {
            return false;
        }

        this.program = new Program(name, outName);
        return this.program;
    }

    runProgram(opData: OpData) {
        const tensorDataList = opData.tensorData;
        const tensorMap = new Map<string, Tensor>();

        const tensorDatas = tensorDataList as Value[];
        for (let i = 0, len = tensorDatas.length; i < len; i++) {
            // @ts-ignore
            const tensorObj = new Obj(tensorDatas[i]);
            const opTensor = new Tensor(tensorObj);
            const tensorName = opTensor.tensorName;
            const tensorId = opTensor.name;
            if ((!opTensor.data || !opTensor.data.length) && this.dataMap.get(tensorId)) {
                opTensor.data = this.dataMap.get(tensorId);
            }
            tensorMap.set(tensorName, opTensor);
        }
        // @ts-ignore

        opData.program.forEach((curProgram: Program) => {
            try {
                const result = curProgram.main(tensorMap, opData.data);
                this.dataMap.set(curProgram.outName, result);
            }
            catch (e) {
            }
        });
    }

    async read({ name }) {
        const data = this.dataMap.get(env.get('ns').layerName || name);
        return data ? Array.from(data) : [];
    }

    dispose() {
        this.dataMap = null;
    }
}
