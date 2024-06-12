import struct
import typing
import dataclasses

# taken from voxie to read the file header
bamFields = [
    ('12s', 'Filename'),
    ('I', 'Rows'),
    ('I', 'Columns'),
    ('I', 'AngularSteps'),
    ('i', 'AngularSteps180'),
    ('I', 'Slices'),
    ('I', 'NumberOfTranslations'),
    ('I', 'NumberOfIntermediateAngles'),
    ('I', 'NumberOfMarginPoints'),
    ('I', 'NumberOfDetectors'),
    ('I', 'BytesPerPixel'),
    ('I', 'NumberOfDiodesPerDetector'),
    ('24x', None),
    ('f', 'MinimumAttenuationCoefficient'),
    ('f', 'MaximumAttenuationCoefficient'),
    ('f', 'TotalNumberOfPhotons'),
    ('f', 'MeasurementTimePerPoint'),
    ('f', 'VelocityNumber'),
    ('f', 'StartAngle'),
    ('f', 'ScanCentrePoint'),
    ('f', 'ScanLengthWithoutRamp'),
    ('f', 'VoxelSize'),
    ('f', 'StageElevation'),
    ('f', 'ElevationIncrement'),
    ('f', 'SOD'),
    ('f', 'SDD'),
    ('f', 'SourceElevation'),
    ('f', 'SourceCentre'),
    ('f', 'SourceDistance'),
    ('f', 'DetectorElevation'),
    ('f', 'DetectorCentre'),
    ('f', 'DetectorDistance'),
    ('f', 'SpacerElevation'),
    ('f', 'ObjectWeight'),
    ('f', 'BeamElevation'),
    ('f', 'CollimatorWidth'),
    ('f', 'CollimatorHeight'),
    ('f', 'AngularStepSizeBetweenImages'),
    ('f', 'PCDClearTimePerPoint'),
    ('f', 'DensityCorrectionFactor'),
    ('f', 'ROICentre'),
    ('f', 'ROIDistance'),
    ('4x', None),
    ('8s', 'SourceType'),
    ('8s', 'SourceEnergy'),
    ('8s', 'SourceIntensity'),
    ('8s', 'DetectorType'),
    ('80s', 'SampleName'),
    ('4s', 'ProgramID'),
    ('16s', 'MeasurementStartTime'),
    ('16s', 'MeasurementStopTime'),
    ('16s', 'TimeAndDateOfLastEdit'),
    ('12s', 'LookUpTableFile1'),
    ('12s', 'LookUpTableFile2'),
    ('12s', 'LookUpTableFile3'),
    ('12s', 'TubeFilter'),
    ('96s', 'ProcessingSteps'),
    ('4x', None),
]

structFormat = '@'
fieldNames = ''
fieldDefaults = []
fields = []
offset = 0
for ty, name in bamFields:
    structFormat += ty
    # print(name, offset, struct.calcsize(ty))
    offset += struct.calcsize(ty)
    if name is not None:
        defVal = struct.unpack(ty, struct.calcsize(ty) * b'\0')[0]
        fieldNames += name + ' '
        fieldDefaults.append(defVal)
        fields.append((name, typing.Any, dataclasses.field(default=defVal)))
bamHeaderStruct = struct.Struct(structFormat)
BamHeader = dataclasses.make_dataclass('BamHeader', fields)

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def get_sinogram(filepath: Path) -> np.ndarray:
    with open(filepath, "rb") as f:
        header_data = f.read(512)
        header = BamHeader(*bamHeaderStruct.unpack(header_data))

        ty = header.Filename[10:11]
        if ty == b'c':
            dtype = np.uint8
        elif ty == b's':
            dtype = np.uint16
        elif ty == b'i':
            dtype = np.uint32
        elif ty == b'r':
            dtype = np.float32
        else:
            raise ValueError('Got unknown data type {!r}'.format(ty))

        imageCount = header.AngularSteps
        imageShape = (header.Columns, header.Rows // imageCount)
        rowSize = header.Columns * header.BytesPerPixel
        offset = (512 + rowSize - 1) // rowSize * rowSize
        dataSize = imageShape[0] * imageShape[1] * header.BytesPerPixel

    sinogram = np.empty(shape=(imageCount, imageShape[0], imageShape[1]), dtype=dtype)

    with open(filepath, "rb") as f:
        for i in range(imageCount):
            f.seek(offset + dataSize * i)
            projection_data = f.read(dataSize)
            projection = np.frombuffer(projection_data, dtype=dtype).reshape(imageShape)
            sinogram[i] = projection

    return sinogram


if __name__ == "__main__":
    INPUT_PATH = Path("/lhome/clarkcs/aRTist_simulations/aRTist_train_data")
    OUTPUT_PATH = Path("/lhome/clarkcs/aRTist_simulations/aRTist_train_data/sinograms")
    for i in range(1000):
        print(i)
        sinogram = get_sinogram(INPUT_PATH / Path(f"train_full_{i}.dd"))
        np.save(OUTPUT_PATH / f"train_full_sinogram_{i:03}", sinogram)