from paraview.simple import *
import sys
import os


def convertFile(inputFile, outputFile, thickness):
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'ExodusIIReader'
    ibeame = ExodusIIReader(FileName=[inputFile])

    # create a new 'Extract Surface'
    extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=ibeame)


    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='Calculator1', Input=extractSurface1)

    # Properties modified on calculator1
    calculator1.ResultArrayName = 'Thickness'
    calculator1.Function = thickness

    # save data
    SaveData(outputFile,  PointDataArrays=['Thickness'],
        FileType='Ascii')
        
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # update the view to ensure updated data information
    renderView1.Update()


if __name__ == "__main__":
    import getopt
    thickness = '0.1'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:f:o:", [
            "thickness=", "file=", "output="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            filename = arg
        if opt in ("-o", "--output"):
            outputName = arg
        if opt in ("-t", "--threshold"):
            thickness = arg
        if opt in ("-h"):
            printHelp()
            sys.exit()
    convertFile(filename, outputName, thickness)
