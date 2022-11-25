import bpy

class Test(bpy.types.Panel):
    bl_label = "Test panel"
    bl_idname = "PT_TestPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = 'UI'
    
    def draw(self, context):
        layout = sef.layout
        
        row = layout.row()
        row.label(text="Sample text", icon = 'cube')
        
        
        
        
def register():
    bpy.utils.register_class(TestPanel)
    
def unregister():
    bpy.utlis.unregister_class(TestPanel)
    
if __name__ == "__main__":
    register()