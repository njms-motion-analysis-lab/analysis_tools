GRADIENTS: str = 'gradients'
POSITIONAL: str = 'positional'


class Wrist:
    def name() -> str:
        'wrist'

    class Position:
        class Left:
            class SensorA:
                DIMENSIONS = {
                    'X':'lwra_x',
                    'Y':'lwra_y',
                    'Z':'lwra_z',
                }

            class SensorB:
                DIMENSIONS = {
                    'X':'lwrb_x',
                    'Y':'lwrb_y',
                    'Z':'lwrb_z',
                }
                
                    
        class Right:
            class SensorA:
                DIMENSIONS = {
                    'X':'rwra_x',
                    'Y':'rwra_y',
                    'Z':'rwra_z',
                }

            class SensorB:
                DIMENSIONS = {
                    'X':'rwrb_x',
                    'Y':'rwrb_y',
                    'Z':'rwrb_z',
                }




PARTS = {
    'wrist': 'Wrist'
}

SIDES = {
    'r' : 'Right',
    'l' : 'Left',
}

SENSORS = {
    'a': 'SensorA',
    'b': 'SensorB',
}