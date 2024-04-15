#!/home/eratr/AUR/paraview/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/bin pvpython

# dependencies; paraview, pyyaml, Pillow

import yaml
import os
import subprocess


from PIL import Image, ImageDraw, ImageFont
import paraview.simple as pv

def main():
    base_yaml_file = '../jobs/sphere.yaml'
    tmp_yaml_file  = 'tmp.yaml'
    shopt_program  = './main'

    parameter_1      = 'default_thickness'
    parameter_1_vals = [0.1,0.05,0.01,0.001]
    parameter_2      = 'filter_radius'
    parameter_2_vals = [0.1,0.25,0.5,1.0,2.0]

    # Read base yaml 
    print("Reading base file from ", base_yaml_file)
    with open(base_yaml_file, 'r') as file:
        base_config = yaml.safe_load(file)

    offset_x = 100
    offset_y = 100
    canvas_size_x = offset_x + 1100 * len(parameter_1_vals)
    canvas_size_y = offset_y + 1100 * len(parameter_2_vals)
    canvas = Image.new('RGB',(canvas_size_x,canvas_size_y),color='white')

    # fnt  = ImageFont.truetype('/home/eratr/AUR/roboto/Roboto-Black.ttf',size=10)
    fnt = ImageFont.load_default()

    for p1i,p1v in enumerate(parameter_1_vals):
        for p2i,p2v in enumerate(parameter_2_vals):
            config = base_config
            config['shell-opt'][parameter_1] = p1v
            config['shell-opt'][parameter_2] = p2v
            exodus_filename = 'tmp_' + to_string(p1v) + '_' + to_string(p2v) + '.e'
            config['shell-opt']['output_file'] = exodus_filename
            
            # write tmp yaml config
            with open(tmp_yaml_file, 'w') as file:
                yaml.dump(config, file)

            # run the command
            run_command = shopt_program + ' -c ' + tmp_yaml_file
            print('Running: [',run_command,']', parameter_1+'=', p1v ,parameter_2+'=', p2v )
            result = subprocess.run([shopt_program, '-c', tmp_yaml_file],stdout=subprocess.PIPE)
            output_lines = str(result.stdout.decode("utf-8")).split('\n')
            last_line = output_lines[-2]
            compliance_str = last_line.split(':')[-1]

            # take screenshot and write to disk
            base_and_ext = os.path.splitext(exodus_filename)
            image_file = base_and_ext[0] + '.png'
            take_screenshot(exodus_filename, image_file)

            # load screenshot, write compliance and add to canvas
            with Image.open(image_file) as img:
                img.load()
            img_draw = ImageDraw.Draw(img)
            cmp_txt = 'c=' + compliance_str
            img_draw.text((0, 0), cmp_txt, font=fnt, fill=(255, 0, 0))
            canvas.paste(img, (offset_x + p1i*1100, offset_y + p2i*1100))

    draw = ImageDraw.Draw(canvas)
    for p1i,p1v in enumerate(parameter_1_vals):
        text = parameter_1 + ' =' + str(p1v)
        draw.text((offset_x + p1i*1100, 0), text, font=fnt, fill=(0, 0, 0))

    for p2i,p2v in enumerate(parameter_2_vals):
        text = parameter_2 + '=' + str(p2v)
        draw.text((0, offset_y + (p2i)*1100), text, font=fnt, fill=(0, 0, 0), direction='ttb')

    file_base_and_ext = os.path.splitext(base_yaml_file)
    canvas_file = file_base_and_ext[0] + '_' + parameter_1 + '_' + parameter_2 + '.png'
    canvas.save(canvas_file)




def take_screenshot(input_file, image_file):
    renderView = pv.GetActiveViewOrCreate('RenderView')
    file = pv.ExodusIIReader(registrationName=input_file,FileName=input_file)
    file.PointVariables = ['p', 'u', 'x']
    file.ElementVariables = ['thickness', 'vm_stress']
    warpByVector = pv.WarpByVector(registrationName='WarpByVector1', Input=file)
    warpByVector.Vectors = ['POINTS', 'p']
    warpByVector.ScaleFactor = 1.0
    
    animationScene = pv.GetAnimationScene()
    animationScene.GoToLast()

    display = pv.Show(warpByVector, renderView, 'UnstructuredGridRepresentation')
    display.Representation = 'Surface'

    renderView.CameraPosition = [0.8664055816863631, 0.3162706741526064, 0.4884753082117011]
    renderView.CameraFocalPoint = [0.25000000000000056, 0.24443628825247246, 0.013463028706610033]
    renderView.CameraViewUp = [-0.5970427255843529, -0.12091182978611986, 0.7930449629400463]
    renderView.CameraParallelScale = 0.5
    renderView.ResetCamera()
    renderView.Update()

    pv.SaveScreenshot(image_file, renderView, ImageResolution=[1000, 1000],OverrideColorPalette='GrayBackground')

    # cleanup to not include in next screenshot
    pv.Delete(warpByVector)
    pv.Delete(file)
    del warpByVector
    del file


def to_string(v):
    return str(v).replace('.ply', '').replace('.', '').replace(',', '').replace('*', '').replace('/', '')

# Wtf is this pythonese?
if __name__=="__main__":
    main()