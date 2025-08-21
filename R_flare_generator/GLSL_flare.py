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
        
        # --- [已修正] GLSL代码现在将效果分离计算，最后合并 ---
        fragment_shader = """
            #version 330 core
            uniform vec2 u_resolution;
            uniform vec2 u_light_pos;
            uniform float u_time;
            uniform vec3 u_light_color;
            uniform float u_flare_size;
            uniform sampler2D u_noise_texture;

            // --- 新的、更符合逻辑的开关 ---
            uniform bool u_generate_main_glow;
            uniform bool u_generate_reflections;

            out vec4 fragColor;
            
            vec4 noise(vec2 p){ return texture(u_noise_texture, p / textureSize(u_noise_texture, 0)); }
            vec4 noise(float p){ return texture(u_noise_texture, vec2(p/textureSize(u_noise_texture, 0).x, 0.0));}

            vec3 flare(vec2 uv, vec2 pos, float seed, float size) {
                vec4 gn = noise(seed-1.0); gn.x = size; 
                vec2 p = pos; 
                vec2 d = uv-p;
                
                // --- 使用独立的变量来存储不同的效果层 ---
                vec3 main_glow_color = vec3(0.0);
                vec3 reflections_color = vec3(0.0);

                // --- Part 1: 主光辉 (光源核心 + 光晕) ---
                if (u_generate_main_glow) {
                    // 计算光源核心
                    main_glow_color += (0.01+gn.x*.2)/(length(d)+0.001);
                    // 在光源核心的基础上，叠加光晕
                    main_glow_color += vec3(noise(atan(d.y,d.x)*256.9+pos.x*2.0).y*.25) * main_glow_color;
                }

                float fltr = length(uv); fltr = (fltr*fltr)*.5+.5; fltr = min(fltr,1.0);

                // --- Part 2: 反射炫光 (完全独立计算) ---
                if (u_generate_reflections) {
                    for (float i=.0; i<20.; i++) {
                        vec4 n = noise(seed+i); vec4 n2 = noise(seed+i*2.1); vec4 nc = noise(seed+i*3.3); nc+=vec4(length(nc)); nc*=.65;
                        for (int j=0; j<3; j++) {
                            float ip = n.x*3.0+float(j)*.1*n2.y*n2.y*n2.y; float is = n.y*n.y*4.5*gn.x+.1; float ia = (n.z*4.0-2.0)*n2.x*n.y;
                            vec2 iuv = (uv*(mix(1.0,length(uv),n.w*n.w)))*mat2(cos(ia),sin(ia),-sin(ia),cos(ia));
                            vec2 id = mix(iuv-p,iuv+p,ip);
                            // 将计算结果累加到独立的 'reflections_color' 变量中
                            if (j == 0) reflections_color.r += pow(max(.0,is-(length(id))),.45)/is*.1*gn.x*nc.r*fltr;
                            if (j == 1) reflections_color.g += pow(max(.0,is-(length(id))),.45)/is*.1*gn.x*nc.g*fltr;
                            if (j == 2) reflections_color.b += pow(max(.0,is-(length(id))),.45)/is*.1*gn.x*nc.b*fltr;
                        }
                    }
                }
                
                // --- 最后一步：像图层一样，将所有效果合并 ---
                return main_glow_color + reflections_color;
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

    def generate(self, light_pos, noise_image_path, time=0.0, flare_size=0.15, light_color=(1.0, 1.0, 1.0),
                 generate_main_glow=False, generate_reflections=True):
        """
        生成一张炫光图像并将其作为PIL Image对象返回。
        :param generate_main_glow: 是否生成光源核心及其光晕。
        :param generate_reflections: 是否生成反射炫光/鬼影。
        :return: PIL.Image.Image 对象
        """
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
        
        # 将Python的布尔值传递给新的GLSL开关
        self.program['u_generate_main_glow'].value = generate_main_glow
        self.program['u_generate_reflections'].value = generate_reflections
        
        self.vao.render(moderngl.TRIANGLE_STRIP)
        image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
        noise_texture.release()
        return image

# --- 如何使用 (已更新为展示修正后的效果) ---
if __name__ == '__main__':
    OUTPUT_RESOLUTION = (1280, 720)
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare_fixed_test'

    if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.isdir(TEXTURE_SOURCE_DIR): raise FileNotFoundError(f"错误：找不到图片源文件夹 '{TEXTURE_SOURCE_DIR}'。")
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_textures: raise FileNotFoundError(f"错误：文件夹 '{TEXTURE_SOURCE_DIR}' 中没有任何图片文件。")

    generator = FlareGenerator(output_size=OUTPUT_RESOLUTION)

    print(f"\n--- 准备生成修正后的效果对比图 ---")
    
    fixed_source_path = os.path.join(TEXTURE_SOURCE_DIR, random.choice(available_textures))
    fixed_light_pos = (generator.width * 0.8, generator.height * 0.3)
    print(f"所有示例将使用相同的图片源: '{os.path.basename(fixed_source_path)}'")
    
    # 示例1: 默认行为 - 只生成反射炫光 (现在效果会很丰富)
    print("\n[正在生成 1/3] 修正后的默认效果 (只有反射)...")
    img1 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=False, generate_reflections=True)
    img1.save(os.path.join(OUTPUT_DIR, "01_fixed_reflections_only.png"))
    print(" -> 已保存: 01_fixed_reflections_only.png")

    # 示例2: 只生成主光辉
    print("\n[正在生成 2/3] 只有主光辉 (光源+光晕)...")
    img2 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=False)
    img2.save(os.path.join(OUTPUT_DIR, "02_main_glow_only.png"))
    print(" -> 已保存: 02_main_glow_only.png")

    # 示例3: 所有效果全开 (效果与原始版本一致)
    print("\n[正在生成 3/3] 所有效果全开 (主光辉+反射)...")
    img3 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=True)
    img3.save(os.path.join(OUTPUT_DIR, "03_all_effects_on.png"))
    print(" -> 已保存: 03_all_effects_on.png")

    print("\n--------------------")
    print(f"成功生成 3 张对比图像！")
    print(f"请在 '{OUTPUT_DIR}' 文件夹中查看 '01_fixed_reflections_only.png'，")
    print("您会发现即使没有光源，反射炫光现在也非常丰富和明亮。")
    print("--------------------")