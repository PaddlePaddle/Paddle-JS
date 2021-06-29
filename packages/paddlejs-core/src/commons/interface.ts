import type Transformer from '../transform/transformer';

export interface ModelOp {
    type: string;
    attrs?: OpAttrs;
    'sub-attrs'?: OpAttrs[];
    inputs: OpInputs;
    outputs: OpOutputs;
}

export interface ModelVar {
    name: string;
    shape: number[];
    data?: number[] | Float32Array;
    persistable?: boolean;
    tensorName?: string;
    total?: number;
    runtime?: number;
    key: string;
}

export interface Model {
    chunkNum?: number;
    ops: ModelOp[];
    vars: ModelVar[];
    multiOutputs?: ModelVar[]
}

export interface ModelConfig {
    modelPath: string;
    modelName?: string;
    feedShape: {
        fw: number;
        fh: number;
    };
    targetSize?: {
        height: number;
        width: number;
    };
    chunkNum?: number;
    fill?: string; // 缩放后用什么颜色填充不足方形部分
    mean?: number[];
    std?: number[];
    bgr?: boolean;
    scale?: number;
    type?: GraphType; // model type
    needPreheat?: boolean;
    plugins?: { // tranform graph plugins
        preTransforms?: Transformer[]; // before creating graph
        transforms?: Transformer[]; // while traversing the ops map
        postTransforms?: Transformer[]; // after creating graph
    };
}
export interface OpInputs {
    [key: string]: any;
}

export interface OpOutputs {
    [key: string]: any;
}

export interface OpAttrs {
    [key: string]: any
}

export interface OpExecutor {
    id: string;
    type: string;
    inputs: OpInputs;
    outputs: OpOutputs;
    attrs: OpAttrs;
    subAttrs: OpAttrs[];
    next: string | null;
    opData: OpData;
    isPacked: boolean;
    finish: boolean;
    inputsName: string[];
    outputsName: string[];
    execute: Function;
}

interface Behavior {
    [key: string]: Function;
}

export interface OpInfo {
    params?: string[] | Function;
    main?: Function | string;
    mainFunc?: Function;
    textureFuncConf?: {
        [key: string]: string[];
    },
    name?: string;
    conf?: object;
    main_packed?: string;
    behavior?: Behavior;
    behaviors?: string[];
}

export interface Ops {
    // key => backend_name
    [key: string]: OpInfo;
}

export interface OpData {
    name: string;
    realName: string;
    isPackedOp: boolean;
    input: OpInputs;
    output: OpOutputs;
    attrs: any;
    subAttrs: object[];
    data: AttrsData;
    inputTensors: any;
    outputTensors: any;
    fShaderParams: any;
    vars: ModelVar[];
    iLayer: number;
    program: any;
    renderData: any[];
    tensorData: any[];
    modelName: string;
}


export interface AttrsData {
    [key: string]: any
}


export interface InputFeed {
    data: Float32Array | number[];
    shape: number[];
    name: string;
}


export enum GraphType {
    SingleOutput = 'single',
    MultipleOutput = 'multiple',
    MultipleInput = 'multipleInput'
}