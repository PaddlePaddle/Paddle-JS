/**
 * @file webgl backend
 * @author yueshuangyan
 */

import { PaddlejsBackend, env } from '@paddlejs/paddlejs-core';
import { ModelVar, OpData, Query } from './types';
import { GLHelper, EShaderType } from './webgl/WebGLUtils';
import { GLTexture, TextureConfig } from './webgl/WebGLTexture';
import { vShaderSource, vShaderData } from './ops/vShader';
import buildShader from './webgl/buildShader';
import GLProgram from './webgl/WebGLProgram';
import { getSizeFromShape, nhwc2nchw } from './utils/dataProcess';
import queryProcess from './utils/queryProcess';


export default class WebGLBackend extends PaddlejsBackend {
    gl: WebGLRenderingContext;
    glVersion: number;
    program?: GLProgram | null;
    frameBuffer?: WebGLFramebuffer | null;
    pbo?: WebGLBuffer | null;
    vertexBuffer?: WebGLBuffer | null;
    textureConf: TextureConfig | null;
    prevTexture?: WebGLTexture | null;
    nextTexture?: WebGLTexture | null;
    currentTexture?: WebGLTexture | null;
    textureBufferIndex?: number;
    cacheTextures: object;
    texturesMap: object;
    uniformLocations: object;
    width_shape_out: number;
    height_shape_out: number;
    width_texture_out: number;
    height_texture_out: number;
    channel: number;
    total_shape: number;
    vShader?: WebGLShader;
    MAX_TEXTURE_SIZE: number;
    queryList: Query[];

    constructor() {
        super();
        this.gl = GLHelper.getWebGLRenderingContext();
        this.textureConf = GLTexture.getTextureConfig(this.gl);
        this.glVersion = GLHelper.getWebglVersion();
        this.MAX_TEXTURE_SIZE = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE);

        // 计算texture cache
        this.cacheTextures = {};
        this.uniformLocations = {};
        // texture buffer
        this.texturesMap = {};

        this.queryList = [];

        // 上一个texture
        this.prevTexture = null;
        // 当前op输出texture
        this.currentTexture = null;

        this.width_shape_out = 1;
        this.height_shape_out = 1;
        this.width_texture_out = 1;
        this.height_texture_out = 1;
        this.channel = 0;
        this.total_shape = 0;

    }

    async init() {
        // 初始化webgl环境
        const gl = this.gl;
        // 关闭相关功能
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.STENCIL_TEST);
        gl.disable(gl.BLEND);
        gl.disable(gl.DITHER);
        gl.disable(gl.POLYGON_OFFSET_FILL);
        gl.disable(gl.SAMPLE_COVERAGE);
        gl.enable(gl.SCISSOR_TEST);
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);

        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vShaderData, gl.STATIC_DRAW);
        // 生成vertextShader
        this.vShader = GLHelper.initShader(gl, EShaderType.VS_SHADER, vShaderSource[this.glVersion - 1]);
        // 帧缓存
        this.frameBuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
        // pbo
        this.pbo = gl.createBuffer();
    }



    createProgram({ name, outTensor, inputTensors, shaderParams, runtime, isFinalOp, isPacked }) {
        // genFscode  buildShader
        const fsCode = buildShader({
            textureConf: this.textureConf,
            type: name,
            inputTensors: inputTensors,
            fShaderParams: shaderParams,
            runtime: runtime,
            isPacked: isPacked,
            key: outTensor.key
        });

        const programInstance = new GLProgram(this.gl, this.vShader as WebGLShader, fsCode, outTensor);
        programInstance.fsCode = fsCode;

        const outTexture = GLTexture.genOutputTexture(this.gl, this.textureConf, outTensor, isFinalOp);

        this.texturesMap[outTensor.tensorId] = outTexture;

        this.program = programInstance;
        return programInstance;
    }

    runProgram(opData: OpData, isRendered: boolean) {
        let query = queryProcess.beginQuery(this.gl, this.glVersion);

        const isPacked = opData.isPackedOp;
        // 设置gpu参数
        opData.program.forEach((program: GLProgram, index) => {
            const outTensor = opData.outputTensors[index];
            const tensorId = outTensor.tensorId;
            this.setOutProps(outTensor);

            // 生成帧缓存材质
            this.attachFrameBuffer(tensorId);
            program.setProgram(this.gl, this.vertexBuffer, isRendered);
            this.program = program;

            this.render(opData.inputTensors, opData.iLayer, isRendered, index, isPacked, opData.modelName);
        });

        if (query) {
            this.queryList.push({ name: opData.name, query, count: 1 });
            query = queryProcess.endQuery(this.gl, this.glVersion, query);
        }
    }

    async read(fetchInfo: ModelVar): Promise<number[]> {
        const pbo = this.createPBO();
        await this.createAndWaitForFence();
        const result = this.downloadFloat32TensorFromBuffer(pbo);

        let shape = fetchInfo ? fetchInfo.shape : [];
        if (env.get('webgl_pack_output')) {
            return result.slice(0, getSizeFromShape(shape));
        }

        shape = env.get('debug') && env.get('shape')
            ? env.get('shape')
            : (this.program as GLProgram).shape;

        const [N, C, H, W] = shape;
        const nhwcFetchShape = [N, H, W, C];
        return nhwc2nchw(result, nhwcFetchShape);
    }

    createPBO() {
        const textureConf = this.textureConf as TextureConfig;
        if (this.glVersion === 2) {
            const gl2 = this.gl as any;
            const buffer = this.pbo;
            gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
            const sizeBytes = 4 * 4 * this.width_texture_out * this.height_texture_out;
            gl2.bufferData(gl2.PIXEL_PACK_BUFFER, sizeBytes, gl2.STREAM_READ);
            gl2.readPixels(0, 0, this.width_texture_out, this.height_texture_out, gl2.RGBA, gl2.FLOAT, 0);
            gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);
            return buffer;
        }

        let downloadData;
        const gl2 = this.gl;
        let textureType = gl2.FLOAT;
        if (textureConf.isFloatTextureReadPixelsEnabled) {
            downloadData = new Float32Array(this.width_texture_out * this.height_texture_out * 4);
        }
        else {
            downloadData = new Uint8Array(this.width_texture_out * this.height_texture_out * 4);
            textureType = gl2.UNSIGNED_BYTE;
        }
        gl2.readPixels(
            0,
            0,
            this.width_texture_out,
            this.height_texture_out,
            gl2.RGBA,
            textureType,
            downloadData);
        if (textureConf.isFloatTextureReadPixelsEnabled) {
            return downloadData;
        }
        const newBuffer = new Float32Array(downloadData.buffer);
        return newBuffer;

    }

    async createAndWaitForFence() {
        const gl2 = this.gl as any;
        const isFenceEnabled = (gl2.fenceSync != null);
        let isFencePassed = () => true;
        if (isFenceEnabled) {
            const sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
            gl2.flush();
            isFencePassed = () => {
                const status = gl2.clientWaitSync(sync, 0, 0);
                return status === gl2.ALREADY_SIGNALED
                    || status === gl2.CONDITION_SATISFIED;
            };
        }
        return new Promise(resolve => {
            this.pollItem(isFencePassed, resolve);
        });
    }

    pollItem(isDone, resolveFn) {
        const fn = () => {
            if (isDone()) {
                resolveFn();
                return;
            }
            setTimeout(fn, 1);
        };
        fn();
    }

    downloadFloat32TensorFromBuffer(buffer) {
        const gl2 = this.gl as any;
        const size: number = 4 * this.width_texture_out * this.height_texture_out;
        if (this.glVersion === 2) {
            const pixels = new Float32Array(size);
            gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
            gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, pixels);
            gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);

            const result: number[] = [];
            if (env.get('webgl_pack_output')) {
                return Array.from(pixels);
            }
            for (let i = 0; i < this.width_texture_out * this.height_texture_out; i++) {
                result.push(pixels[4 * i]);
            }
            return result;
        }

        const pixels = buffer;
        const result = [] as number[];
        for (let i = 0; i < this.width_texture_out * this.height_texture_out; i++) {
            const index = (this.textureConf as TextureConfig).isFloatTextureReadPixelsEnabled ? 4 * i : i;
            result.push(pixels[index]);
        }
        return result;

    }

    setOutProps(outTensor?) {
        const {
            width_shape = 1,
            height_shape = 1,
            width_texture = 1,
            height_texture = 1,
            channel = 0,
            total_shape = 0
        } = outTensor;

        this.width_shape_out = width_shape;
        this.height_shape_out = height_shape;
        this.width_texture_out = width_texture;
        this.height_texture_out = height_texture;
        this.channel = channel;
        this.total_shape = total_shape;
    }

    attachFrameBuffer(tensorId: string) {
        this.prevTexture = this.currentTexture;
        this.currentTexture = this.texturesMap[tensorId];

        const gl = this.gl;
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER, // The target is always a FRAMEBUFFER.
            gl.COLOR_ATTACHMENT0, // We are providing the color buffer.表示texture是颜色关联对象
            gl.TEXTURE_2D, // This is a 2D image texture.
            this.currentTexture as WebGLTexture, // The texture.
            0 // 0, we aren't using MIPMAPs
        );
        gl.viewport(
            0,
            0,
            this.width_texture_out,
            this.height_texture_out
        );
        gl.scissor(
            0,
            0,
            this.width_texture_out,
            this.height_texture_out
        );
        return this.frameBuffer;
    }

    render(
        data: any = [],
        iLayer: number = 0,
        isRendered: Boolean = false,
        index: number,
        isPacked: Boolean = false,
        modelName: string
    ) {
        const gl = this.gl;
        const that = this;
        let textureIndex = 0;
        data.forEach(item => {
            const loc = that.getUniformLoc('texture_' + item.name, iLayer, isRendered, index, modelName);
            if (!loc) {
                return;
            }
            that.initTexture(textureIndex, item, iLayer, isRendered, isPacked, modelName);
            gl.uniform1i(loc, textureIndex++);
        });
        // gl.clearColor(.0, .0, .0, 1);
        // gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    initTexture(index, item, iLayer, isRendered, isPacked, modelName) {
        const gl = this.gl;
        const textureConf = this.textureConf as TextureConfig;
        const tensorName = item.opts.type;
        const prefix = modelName + '_';
        let texture;
        if (!item.data) {
            // texture = this.prevTexture;
            texture = this.texturesMap[item.opts.type];
        }
        else {
            // texture = gl.createTexture();
            if (isRendered && (iLayer > 1 || (iLayer === 1 && !tensorName.endsWith('_image')))) {
                const tData = this.cacheTextures[prefix + iLayer];
                texture = tData['texture_' + tensorName];
            }
            else {
                texture = gl.createTexture();
                this.cacheTextures[prefix + iLayer] = this.cacheTextures[prefix + iLayer] || {};
                this.cacheTextures[prefix + iLayer]['texture_' + tensorName] = texture;
            }
        }

        gl.activeTexture(gl[`TEXTURE${index}`]);
        gl.bindTexture(gl.TEXTURE_2D, texture);

        if (item.data && (!isRendered || (isRendered && iLayer === 1 && tensorName.endsWith('_image')))) {
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            if (this.glVersion === 2) {
                const useHalfFloat = env.get('webgl_force_half_float_texture');
                const internalFormat = isPacked
                    ? useHalfFloat
                        ? textureConf.internalFormatPackedHalfFloat
                        : textureConf.internalFormatPacked
                    : useHalfFloat
                        ? textureConf.internalFormatHalfFloat
                        : textureConf.internalFormat;

                const textureFormat = isPacked ? gl.RGBA : textureConf.textureFormat;

                gl.texImage2D(gl.TEXTURE_2D,
                    0,
                    internalFormat,
                    item.width_texture,
                    item.height_texture,
                    0,
                    textureFormat,
                    gl.FLOAT,
                    item.data
                );
            }
            else {
                const temp = new Float32Array(item.width_texture * item.height_texture * 4);
                for (let i = 0; i < item.data.length; i++) {
                    if (isPacked) {
                        temp[i] = item.data[i];
                    }
                    else {
                        // 填充 r 通道数据，其他通道 为 0
                        temp[i * 4] = item.data[i];
                        temp[i * 4 + 1] = 0;
                        temp[i * 4 + 2] = 0;
                        temp[i * 4 + 3] = 0;
                    }
                }
                gl.texImage2D(gl.TEXTURE_2D,
                    0,
                    gl.RGBA,
                    item.width_texture,
                    item.height_texture,
                    0,
                    gl.RGBA,
                    gl.FLOAT,
                    temp);
            }
        }
    }

    getUniformLoc(name, ilayer, isRendered, index, modelName) {
        const prefix = modelName + '_';
        if (isRendered) {
            return this.uniformLocations[prefix + ilayer][name + index];
        }
        const loc = this.gl.getUniformLocation((this.program as GLProgram).program as WebGLProgram, name);
        // 缓存
        this.uniformLocations[prefix + ilayer] = this.uniformLocations[prefix + ilayer] || {};
        this.uniformLocations[prefix + ilayer][name + index] = loc;
        return loc;
    }

    dispose() {

    }
}
