import unreal
import random
import os
import time

#---------------------------------------------------------------------------------
#--- 用户可配置参数 ---
#---------------------------------------------------------------------------------

NUM_ITERATIONS = 10
BASE_OUTPUT_PATH = "D:/UE5_Datasets/FlareProject"
LEVEL_SEQUENCE_ASSET_PATH = '/Game/NewLevelSequence.NewLevelSequence'

CAM_START_POS_MIN = unreal.Vector(-180.0, -60.0, -40.0)
CAM_START_POS_MAX = unreal.Vector(-180.0, 60.0, 40.0)
CAM_END_POS_MIN = unreal.Vector(-180.0, -60.0, -40.0)
CAM_END_POS_MAX = unreal.Vector(-180.0, 60.0, 40.0)
LIGHT_RADIUS_MIN = 5.0
LIGHT_RADIUS_MAX = 25.0
FLARE_INTENSITY_MIN = 1.0
FLARE_INTENSITY_MAX = 15.0
BLOOM_INTENSITY_MIN = 0.1
BLOOM_INTENSITY_MAX = 1.0

# 视频序列参数
FRAMES_PER_SEQUENCE = 24  # 每个序列的帧数（1秒@24fps）

#---------------------------------------------------------------------------------
#--- 正确的视频序列生成方案 ---
#---------------------------------------------------------------------------------

def get_actor_by_label(actor_label):
    editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = editor_actor_subsystem.get_all_level_actors()
    for actor in actors:
        if actor.get_actor_label() == actor_label:
            return actor
    unreal.log_error(f"找不到标签为 '{actor_label}' 的Actor！")
    return None

def interpolate_vector(start_vec, end_vec, t):
    """线性插值两个向量 (t: 0.0 到 1.0)"""
    return unreal.Vector(
        start_vec.x + (end_vec.x - start_vec.x) * t,
        start_vec.y + (end_vec.y - start_vec.y) * t,
        start_vec.z + (end_vec.z - start_vec.z) * t
    )

def generate_video_sequence(camera, start_pos, end_pos, output_dir, base_name):
    """生成从起点到终点的连续视频序列"""
    unreal.log(f"📹 生成视频序列: {start_pos} -> {end_pos}")
    unreal.log(f"📁 输出到: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    captured_frames = 0
    
    for frame_idx in range(FRAMES_PER_SEQUENCE):
        # 计算插值参数 (0.0 到 1.0)
        t = frame_idx / (FRAMES_PER_SEQUENCE - 1)
        
        # 插值计算当前相机位置
        current_pos = interpolate_vector(start_pos, end_pos, t)
        
        # 设置相机位置
        camera.set_actor_location(current_pos, False, False)
        
        # 稍微等待确保相机位置更新
        time.sleep(0.1)
        
        # 截图
        screenshot_name = f"{base_name}_frame_{frame_idx+1:04d}"
        
        try:
            unreal.AutomationLibrary.take_high_res_screenshot(1920, 1080, screenshot_name)
            captured_frames += 1
            
            if frame_idx % 5 == 0:  # 每5帧报告一次进度
                unreal.log(f"📸 已捕获 {frame_idx+1}/{FRAMES_PER_SEQUENCE} 帧")
                
        except Exception as e:
            unreal.log_error(f"❌ 第 {frame_idx+1} 帧截图失败: {e}")
    
    unreal.log(f"✅ 序列完成：{captured_frames}/{FRAMES_PER_SEQUENCE} 帧")
    return captured_frames

def organize_screenshots_to_sequence(iteration_num):
    """整理截图到视频序列文件夹"""
    screenshots_dir = "D:/Epic Games/flare2/Saved/Screenshots/WindowsEditor"
    
    if not os.path.exists(screenshots_dir):
        unreal.log_error(f"❌ 截图目录不存在: {screenshots_dir}")
        return False
    
    # 获取当前迭代的所有截图
    png_files = [f for f in os.listdir(screenshots_dir) if f.endswith('.png')]
    
    # 分类移动文件
    flare_files = [f for f in png_files if f"flare_{iteration_num:04d}_frame_" in f]
    no_flare_files = [f for f in png_files if f"no_flare_{iteration_num:04d}_frame_" in f]
    
    # 移动带炫光序列
    flare_dir = os.path.join(BASE_OUTPUT_PATH, f"{iteration_num:04d}", "flare")
    os.makedirs(flare_dir, exist_ok=True)
    
    for file_name in sorted(flare_files):
        source = os.path.join(screenshots_dir, file_name)
        target = os.path.join(flare_dir, file_name)
        try:
            import shutil
            shutil.move(source, target)
        except:
            pass
    
    # 移动无炫光序列
    no_flare_dir = os.path.join(BASE_OUTPUT_PATH, f"{iteration_num:04d}", "no_flare")
    os.makedirs(no_flare_dir, exist_ok=True)
    
    for file_name in sorted(no_flare_files):
        source = os.path.join(screenshots_dir, file_name)
        target = os.path.join(no_flare_dir, file_name)
        try:
            import shutil
            shutil.move(source, target)
        except:
            pass
    
    unreal.log(f"📁 已整理第 {iteration_num} 次迭代的截图")
    unreal.log(f"   - 带炫光: {len(flare_files)} 帧")
    unreal.log(f"   - 无炫光: {len(no_flare_files)} 帧")
    
    return len(flare_files) > 0 and len(no_flare_files) > 0

def generate_flare_video_dataset():
    """生成炫光视频数据集"""
    unreal.log("🚀 开始生成炫光视频数据集...")
    
    # 获取场景对象
    camera = get_actor_by_label("AutomationCamera")
    light = get_actor_by_label("AutomationLight")
    ppv = get_actor_by_label("AutomationPPV")
    
    if not all([camera, light, ppv]):
        unreal.log_error("❌ 找不到必要的Actor")
        return False
    
    # 创建基础输出目录
    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)
    
    successful_iterations = 0
    
    for i in range(NUM_ITERATIONS):
        iteration_num = i + 1
        unreal.log(f"")
        unreal.log(f"=== 第 {iteration_num}/{NUM_ITERATIONS} 次迭代 ===")
        
        # 生成随机起始和终点位置
        start_pos = unreal.Vector(
            random.uniform(CAM_START_POS_MIN.x, CAM_START_POS_MAX.x),
            random.uniform(CAM_START_POS_MIN.y, CAM_START_POS_MAX.y),
            random.uniform(CAM_START_POS_MIN.z, CAM_START_POS_MAX.z)
        )
        end_pos = unreal.Vector(
            random.uniform(CAM_END_POS_MIN.x, CAM_END_POS_MAX.x),
            random.uniform(CAM_END_POS_MIN.y, CAM_END_POS_MAX.y),
            random.uniform(CAM_END_POS_MIN.z, CAM_END_POS_MAX.z)
        )
        
        # 生成随机光源和效果参数
        light_radius = random.uniform(LIGHT_RADIUS_MIN, LIGHT_RADIUS_MAX)
        flare_intensity = random.uniform(FLARE_INTENSITY_MIN, FLARE_INTENSITY_MAX)
        bloom_intensity = random.uniform(BLOOM_INTENSITY_MIN, BLOOM_INTENSITY_MAX)
        
        unreal.log(f"🎲 随机参数:")
        unreal.log(f"   起点: {start_pos}")
        unreal.log(f"   终点: {end_pos}")
        unreal.log(f"   光源半径: {light_radius:.2f}")
        unreal.log(f"   炫光强度: {flare_intensity:.2f}")
        unreal.log(f"   泛光强度: {bloom_intensity:.2f}")
        
        # 设置光源
        light_component = light.get_component_by_class(unreal.PointLightComponent)
        if light_component:
            light_component.set_editor_property("source_radius", light_radius)
        
        # 创建临时输出目录（稍后会整理）
        temp_flare_dir = os.path.join(BASE_OUTPUT_PATH, "temp")
        temp_no_flare_dir = os.path.join(BASE_OUTPUT_PATH, "temp")
        
        iteration_success = True
        
        # === 生成带炫光视频序列 ===
        unreal.log("🔥 生成带炫光视频序列...")
        ppv.settings.override_lens_flare_intensity = True
        ppv.settings.lens_flare_intensity = flare_intensity
        ppv.settings.override_bloom_intensity = True
        ppv.settings.bloom_intensity = bloom_intensity
        
        flare_frames = generate_video_sequence(
            camera, start_pos, end_pos, temp_flare_dir, f"flare_{iteration_num:04d}"
        )
        
        if flare_frames < FRAMES_PER_SEQUENCE:
            unreal.log_error("带炫光序列生成不完整")
            iteration_success = False
        
        # 等待截图完成
        unreal.log("⏰ 等待带炫光截图完成...")
        time.sleep(3)
        
        # === 生成无炫光视频序列 ===
        unreal.log("❄️ 生成无炫光视频序列...")
        ppv.settings.lens_flare_intensity = 0.0
        ppv.settings.bloom_intensity = 0.0
        
        no_flare_frames = generate_video_sequence(
            camera, start_pos, end_pos, temp_no_flare_dir, f"no_flare_{iteration_num:04d}"
        )
        
        if no_flare_frames < FRAMES_PER_SEQUENCE:
            unreal.log_error("无炫光序列生成不完整")
            iteration_success = False
        
        # 等待截图完成
        unreal.log("⏰ 等待无炫光截图完成...")
        time.sleep(3)
        
        # 整理截图到正确位置
        if organize_screenshots_to_sequence(iteration_num):
            successful_iterations += 1
            unreal.log(f"✅ 第{iteration_num}次迭代完成")
        else:
            unreal.log_error(f"❌ 第{iteration_num}次迭代整理失败")
    
    unreal.log("")
    unreal.log(f"🎉 视频数据集生成完成：{successful_iterations}/{NUM_ITERATIONS} 次成功")
    unreal.log(f"📁 输出路径: {BASE_OUTPUT_PATH}")
    unreal.log(f"📹 每个序列包含 {FRAMES_PER_SEQUENCE} 帧（1秒@24fps）")
    unreal.log("💡 可以用FFmpeg将图片序列转换为视频:")
    unreal.log("   ffmpeg -i flare_%04d.png -c:v libx264 -pix_fmt yuv420p flare_video.mp4")
    
    return successful_iterations > 0

def main():
    """主函数"""
    unreal.log("🎬 炫光视频数据集生成器")
    unreal.log("🎯 生成从随机起点到终点的连续视频序列")
    unreal.log("")
    
    success = generate_flare_video_dataset()
    
    if success:
        unreal.log("")
        unreal.log("✅ 生成成功！")
        unreal.log("📁 每个迭代包含成对的视频序列（图片帧）")
        unreal.log("🎬 可用于训练视频处理的AI模型")
    else:
        unreal.log("")
        unreal.log("❌ 生成失败")

# 运行主程序
main()