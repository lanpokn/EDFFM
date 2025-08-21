import moderngl
import numpy as np
from PIL import Image
import os
import random

class FlareGenerator:
    """
    一个使用GLSL Shader生成镜头炫光图像的封装类。
    """
    def __init__(self, output_size=(1920, 1080)):
        self.output_size = output_size
        self.width, self.height = output_size
        self.ctx = moderngl.create_standalone_context()

        vertex_shader = """
            #version 330
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        """
        
        fragment_shader = """
            #version 330 core
            uniform vec2 u_resolution;
            uniform vec2 u_light_pos;
            uniform float u_time;
            uniform vec3 u_light_color;
            uniform float u_flare_size;
            uniform sampler2D u_noise_texture;
            out vec4 fragColor;
            
            vec4 noise(vec2 p){ return texture(u_noise_texture, p / textureSize(u_noise_texture, 0)); }
            vec4 noise(float p){ return texture(u_noise_texture, vec2(p/textureSize(u_noise_texture, 0).x, 0.0));}

            vec3 flare(vec2 uv, vec2 pos, float seed, float size) {
                vec4 gn = noise(seed-1.0); gn.x = size; vec3 c = vec3(.0); vec2 p = pos; vec2 d = uv-p;
                c += (0.01+gn.x*.2)/(length(d)+0.001); c += vec3(noise(atan(d.y,d.x)*256.9+pos.x*2.0).y*.25)*c;
                float fltr = length(uv); fltr = (fltr*fltr)*.5+.5; fltr = min(fltr,1.0);
                for (float i=.0; i<20.; i++) {
                    vec4 n = noise(seed+i); vec4 n2 = noise(seed+i*2.1); vec4 nc = noise(seed+i*3.3); nc+=vec4(length(nc)); nc*=.65;
                    for (int j=0; j<3; j++) {
                        float ip = n.x*3.0+float(j)*.1*n2.y*n2.y*n2.y; float is = n.y*n.y*4.5*gn.x+.1; float ia = (n.z*4.0-2.0)*n2.x*n.y;
                        vec2 iuv = (uv*(mix(1.0,length(uv),n.w*n.w)))*mat2(cos(ia),sin(ia),-sin(ia),cos(ia));
                        vec2 id = mix(iuv-p,iuv+p,ip);
                        if (j == 0) c.r += pow(max(.0,is-(length(id))),.45)/is*.1*gn.x*nc.r*fltr;
                        if (j == 1) c.g += pow(max(.0,is-(length(id))),.45)/is*.1*gn.x*nc.g*fltr;
                        if (j == 2) c.b += pow(max(.0,is-(length(id))),.45)/is*.1*gn.x*nc.b*fltr;
                    }
                } return c;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution - 0.5;
                uv.x *= u_resolution.x / u_resolution.y;
                uv *= 2.0;
                vec2 pos = u_light_pos / u_resolution - 0.5;
                pos.x *= u_resolution.x / u_resolution.y;
                pos *= 2.0;
                vec3 color = flare(uv, pos, u_time, u_flare_size) * u_light_color;
                color += noise(gl_FragCoord.xy).xyz * 0.01;
                fragColor = vec4(color, 1.0);
            }
        """

        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
        self.texture_output = self.ctx.texture(self.output_size, 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.texture_output])

    def generate(self, light_pos, noise_image_path, output_path, time=0.0, flare_size=0.15, light_color=(1.0, 1.0, 1.0)):
        noise_img = Image.open(noise_image_path).convert("RGBA")
        noise_texture = self.ctx.texture(noise_img.size, 4, noise_img.tobytes())
        noise_texture.use(0)
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.program['u_resolution'].value = self.output_size
        self.program['u_light_pos'].value = light_pos
        self.program['u_time'].value = time
        self.program['u_light_color'].value = light_color
        self.program['u_flare_size'].value = flare_size
        self.program['u_noise_texture'].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)
        image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
        image.save(output_path)
        # 更新打印信息，使其更简洁
        print(f"  -> 已生成: {os.path.basename(output_path)}")
        noise_texture.release()

# --- 如何使用 (已更新为批量生成模式) ---
if __name__ == '__main__':
    # --- 配置参数 ---
    NUM_TO_GENERATE = 20
    OUTPUT_RESOLUTION = (1280, 720)
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare'

    # --- 准备工作 ---
    # 1. 检查并创建输出目录
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: '{OUTPUT_DIR}'")

    # 2. 检查并加载可用的图片源
    if not os.path.isdir(TEXTURE_SOURCE_DIR):
        raise FileNotFoundError(f"错误：找不到图片源文件夹 '{TEXTURE_SOURCE_DIR}'。")
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_textures:
        raise FileNotFoundError(f"错误：文件夹 '{TEXTURE_SOURCE_DIR}' 中没有任何图片文件。")
    print(f"在 '{TEXTURE_SOURCE_DIR}' 中找到了 {len(available_textures)} 个可用的图片源。")

    # 3. 实例化生成器
    generator = FlareGenerator(output_size=OUTPUT_RESOLUTION)

    # --- 开始批量生成 ---
    print(f"\n--- 准备批量生成 {NUM_TO_GENERATE} 张炫光图像 ---")
    
    for i in range(1, NUM_TO_GENERATE + 1):
        # --- 在每次循环中，完全随机化所有参数 ---

        # 1. 随机选择一个图片源
        source_texture_name = random.choice(available_textures)
        source_path = os.path.join(TEXTURE_SOURCE_DIR, source_texture_name)
        
        # 2. 随机化光源位置 (在画面边界内)
        light_pos_x = random.randint(0, generator.width)
        light_pos_y = random.randint(0, generator.height)
        
        # 3. 随机化炫光大小和颜色
        flare_size = random.uniform(0.05, 0.35)
        # 生成更鲜艳的颜色，避免暗色
        r = random.uniform(0.6, 1.0)
        g = random.uniform(0.6, 1.0)
        b = random.uniform(0.6, 1.0)
        light_color = (r, g, b)

        # 4. 随机化时间种子
        time_seed = random.uniform(0, 500)

        # 5. 定义唯一的输出文件名
        # 格式: 001_from_texture_name.png
        base_texture_name = os.path.splitext(source_texture_name)[0]
        output_filename = f"{i:03d}_from_{base_texture_name}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n[正在生成 {i}/{NUM_TO_GENERATE}] 使用图片源: '{source_texture_name}'")

        # 6. 调用生成函数
        generator.generate(
            light_pos=(light_pos_x, light_pos_y),
            noise_image_path=source_path,
            output_path=output_path,
            time=time_seed,
            flare_size=flare_size,
            light_color=light_color
        )

    print("\n--------------------")
    print(f"成功生成 {NUM_TO_GENERATE} 张炫光图像！")
    print(f"所有文件已保存到 '{OUTPUT_DIR}' 文件夹中。")
    print("--------------------")