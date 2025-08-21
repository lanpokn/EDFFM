import unreal

unreal.log("--- MRQ PNG Setting Property Interrogation Script (v2) ---")

try:
    # Set up a temporary environment to inspect the setting object
    movie_pipeline_subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    pipeline_queue = movie_pipeline_subsystem.get_queue()
    temp_job = pipeline_queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    config = temp_job.get_configuration()
    png_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    
    if png_setting:
        unreal.log("Successfully accessed PNG Setting object. Interrogating properties...")
        
        # Get the class of the object to inspect its properties
        setting_class = png_setting.get_class()
        
        unreal.log(f"--- All available UProperties for '{setting_class.get_name()}' ---")
        
        # [CORRECTED METHOD] Iterate through all properties of the class using the '_fields_' attribute
        properties = setting_class._fields_
        
        if not properties:
            unreal.log_warning("Could not find any properties for this class.")
        else:
            for prop in properties:
                # Get the internal name of the property
                prop_name = prop.get_name()
                unreal.log(f"    -> Found Property: '{prop_name}'")
                
        unreal.log("--- Interrogation Complete ---")
    else:
        unreal.log_error("Failed to create an instance of MoviePipelineImageSequenceOutput_PNG.")
        
    # Clean up the temporary job
    pipeline_queue.delete_job(temp_job)

except Exception as e:
    unreal.log_error(f"Diagnostic script encountered an error: {e}")

unreal.log("--- Diagnostic Script Finished ---")