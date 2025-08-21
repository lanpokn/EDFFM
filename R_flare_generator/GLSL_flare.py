# from https://www.shadertoy.com/view/lsBGDKimport moderngl
import moderngl
import numpy as np
from PIL import Image

class FlareGenerator:
    """
    一个使用GLSL Shader生成镜头炫光图像的封装类。
    """
    def __init__(self, output_size=(1920, 1080)):
        self.output_size = output_size
        self.width, self.height = self.output_size

        # 在Windows上，不需要指定后端，让moderngl自动选择即可
        self.ctx = moderngl.create_standalone_context()

        # 顶点着色器
        vertex_shader = """
            #version 330
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        """
        
        # 片段着色器
        fragment_shader = """
            #version 330 core
            uniform vec2 u_resolution;
            uniform vec2 u_light_pos;
            uniform float u_time;
            uniform vec3 u_light_color;
            uniform float u_flare_size;
            uniform sampler2D u_noise_texture;
            out vec4 fragColor;
            
            // --- [代码修正处] ---
            // 我们需要恢复所有被flare函数用到的noise重载函数
            vec4 noise(vec2 p){ return texture(u_noise_texture, p / textureSize(u_noise_texture, 0)); }
            vec4 noise(float p){ return texture(u_noise_texture, vec2(p/textureSize(u_noise_texture, 0).x, 0.0));}
            // --------------------

            vec3 flare(vec2 uv, vec2 pos, float seed, float size) {
                // ... (完整的flare函数代码粘贴在这里) ...
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
        print(f"炫光图像已生成并保存到: {output_path}")
        noise_texture.release()

# --- 如何使用 ---
if __name__ == '__main__':
    # 为了让脚本能独立运行，我们先程序化地创建一个噪声图
    def create_noise_texture(path='noise.png', size=(256, 256)):
        img_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(path)
        print(f"已创建随机噪声纹理: {path}")

    create_noise_texture()

    # 1. 实例化生成器，可以指定输出分辨率
    generator = FlareGenerator(output_size=(1280, 720))

    # 2. 调用generate函数，像调用一个普通Python函数一样！
    generator.generate(
        light_pos=(900, 300),           # 光源在图像右侧
        noise_image_path='noise.png',
        output_path='flare_output_1.png',
        time=15.5,                      # 不同的时间会产生不同的炫光细节
        flare_size=0.2,                 # 稍大的炫光
        light_color=(1.0, 0.8, 0.6)     # 温暖的橙色光
    )

    generator.generate(
        light_pos=(100, 600),           # 光源在图像左下角
        noise_image_path='noise.png',
        output_path='flare_output_2.png',
        time=30.2,
        flare_size=0.1,                 # 较小的炫光
        light_color=(0.7, 0.8, 1.0)     # 冷色的蓝色光
    )