import json
import argparse
from os.path import join

data = {
    'MOT17': {
        'train': [
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN'
        ],
        'test': [
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN'
        ]
    },
    'MOT20': {
        'test': [
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ],
        'train': [
            'MOT20-01',
            'MOT20-02',
            'MOT20-03',
            'MOT20-05'
        ]
    },
    'KITTI': {
        'train': [
            "0000", "0002", "0004", "0006", "0008", "0010", "0012", "0014", "0016", "0018", "0020",
            "0001", "0003", "0005", "0007", "0009", "0011", "0013", "0015", "0017", "0019"
        ]

    },
    'PersonPath22': {
        'test': [
            "uid_vid_00008.mp4",  "uid_vid_00009.mp4",  "uid_vid_00011.mp4",  "uid_vid_00013.mp4",  "uid_vid_00018.mp4",  "uid_vid_00019.mp4",
            "uid_vid_00020.mp4",  "uid_vid_00024.mp4",  "uid_vid_00028.mp4",  "uid_vid_00030.mp4",  "uid_vid_00031.mp4",  "uid_vid_00035.mp4",
            "uid_vid_00036.mp4",  "uid_vid_00038.mp4",  "uid_vid_00043.mp4",  "uid_vid_00045.mp4",  "uid_vid_00046.mp4",  "uid_vid_00048.mp4",
            "uid_vid_00051.mp4",  "uid_vid_00056.mp4",  "uid_vid_00057.mp4",  "uid_vid_00063.mp4",  "uid_vid_00064.mp4",  "uid_vid_00066.mp4",
            "uid_vid_00067.mp4",  "uid_vid_00068.mp4",  "uid_vid_00069.mp4",  "uid_vid_00071.mp4",  "uid_vid_00076.mp4",  "uid_vid_00078.mp4",
            "uid_vid_00079.mp4",  "uid_vid_00080.mp4",  "uid_vid_00082.mp4",  "uid_vid_00085.mp4",  "uid_vid_00086.mp4",  "uid_vid_00087.mp4",
            "uid_vid_00090.mp4",  "uid_vid_00092.mp4",  "uid_vid_00096.mp4",  "uid_vid_00098.mp4",  "uid_vid_00099.mp4",  "uid_vid_00100.mp4",
            "uid_vid_00102.mp4",  "uid_vid_00105.mp4",  "uid_vid_00107.mp4",  "uid_vid_00109.mp4",  "uid_vid_00113.mp4",  "uid_vid_00114.mp4",
            "uid_vid_00117.mp4",  "uid_vid_00144.mp4",  "uid_vid_00147.mp4",  "uid_vid_00149.mp4",  "uid_vid_00150.mp4",  "uid_vid_00118.mp4",
            "uid_vid_00121.mp4",  "uid_vid_00122.mp4",  "uid_vid_00124.mp4",  "uid_vid_00125.mp4",  "uid_vid_00126.mp4",  "uid_vid_00127.mp4",
            "uid_vid_00130.mp4",  "uid_vid_00133.mp4",  "uid_vid_00141.mp4",  "uid_vid_00153.mp4",  "uid_vid_00158.mp4",  "uid_vid_00161.mp4",
            "uid_vid_00163.mp4",  "uid_vid_00166.mp4",  "uid_vid_00167.mp4",  "uid_vid_00169.mp4",  "uid_vid_00170.mp4",  "uid_vid_00172.mp4",
            "uid_vid_00173.mp4",  "uid_vid_00174.mp4",  "uid_vid_00175.mp4",  "uid_vid_00178.mp4",  "uid_vid_00179.mp4",  "uid_vid_00183.mp4",
            "uid_vid_00189.mp4",  "uid_vid_00190.mp4",  "uid_vid_00191.mp4",  "uid_vid_00193.mp4",  "uid_vid_00198.mp4",  "uid_vid_00200.mp4",  "uid_vid_00201.mp4",
            "uid_vid_00205.mp4",  "uid_vid_00207.mp4",  "uid_vid_00212.mp4",  "uid_vid_00218.mp4",  "uid_vid_00219.mp4",  "uid_vid_00221.mp4",
            "uid_vid_00222.mp4",  "uid_vid_00226.mp4",  "uid_vid_00228.mp4",  "uid_vid_00230.mp4"
            # ,  "uid_vid_00162.mp4",  "uid_vid_00234.mp4",  "uid_vid_00235.mp4" # ignored as its videos is not the same as others
        ]
    },
    'VIRAT-S': {'train': [
        'VIRAT_S_050000_10_001462_001491', 'VIRAT_S_010206_03_000546_000580', 'VIRAT_S_010005_04_000299_000323', 'VIRAT_S_010109_07_000876_000910', 'VIRAT_S_010200_10_000923_000959', 'VIRAT_S_010200_09_000886_000915', 'VIRAT_S_010004_02_000191_000237', 'VIRAT_S_010003_01_000111_000137', 'VIRAT_S_010200_05_000658_000700', 'VIRAT_S_010201_05_000499_000527', 'VIRAT_S_010208_08_000807_000831', 'VIRAT_S_010001_05_000649_000684', 'VIRAT_S_010110_04_000777_000812', 'VIRAT_S_010106_05_000954_000996', 'VIRAT_S_010110_05_000899_000935', 'VIRAT_S_010208_07_000768_000791', 'VIRAT_S_010111_05_000762_000799', 'VIRAT_S_010201_01_000125_000152', 'VIRAT_S_010207_01_000712_000752', 'VIRAT_S_010206_04_000720_000767', 'VIRAT_S_010204_05_000856_000890', 'VIRAT_S_010205_03_000370_000395', 'VIRAT_S_010004_01_000163_000188', 'VIRAT_S_010002_04_000307_000350', 'VIRAT_S_050000_12_001591_001619', 'VIRAT_S_010114_02_000765_000802', 'VIRAT_S_010202_02_000161_000189', 'VIRAT_S_010003_11_000956_000982', 'VIRAT_S_010202_00_000001_000033', 'VIRAT_S_010207_06_001064_001097', 'VIRAT_S_010207_02_000790_000816', 'VIRAT_S_010107_00_000019_000057', 'VIRAT_S_010202_03_000313_000355', 'VIRAT_S_010111_00_000000_000032', 'VIRAT_S_010200_08_000838_000867', 'VIRAT_S_010002_05_000397_000420', 'VIRAT_S_010203_10_001092_001121', 'VIRAT_S_010111_09_000981_001014', 'VIRAT_S_010111_07_000872_000909', 'VIRAT_S_010002_01_000123_000148', 'VIRAT_S_010002_02_000174_000204', 'VIRAT_S_010003_03_000219_000259', 'VIRAT_S_010002_06_000441_000467', 'VIRAT_S_010001_06_000685_000722', 'VIRAT_S_010207_09_001484_001510', 'VIRAT_S_010206_00_000007_000035', 'VIRAT_S_010204_07_000942_000989', 'VIRAT_S_010208_09_000857_000886', 'VIRAT_S_010001_03_000537_000563', 'VIRAT_S_010207_08_001308_001332', 'VIRAT_S_010205_04_000545_000576', 'VIRAT_S_010106_01_000493_000526', 'VIRAT_S_010005_06_000475_000499', 'VIRAT_S_010204_06_000913_000939', 'VIRAT_S_010201_09_000770_000801', 'VIRAT_S_010003_05_000499_000523', 'VIRAT_S_010002_07_000522_000547', 'VIRAT_S_010208_02_000150_000180', 'VIRAT_S_010005_02_000177_000203', 'VIRAT_S_010113_07_000965_001013', 'VIRAT_S_010207_04_000929_000954', 'VIRAT_S_010005_00_000048_000075', 'VIRAT_S_010115_02_000485_000516', 'VIRAT_S_010201_02_000167_000197', 'VIRAT_S_050203_06_001202_001264', 'VIRAT_S_010111_06_000820_000860', 'VIRAT_S_010203_03_000400_000435', 'VIRAT_S_010001_09_000921_000952', 'VIRAT_S_010113_02_000434_000479', 'VIRAT_S_010000_06_000728_000762', 'VIRAT_S_010207_05_001013_001038', 'VIRAT_S_010204_00_000030_000059', 'VIRAT_S_010005_05_000397_000430', 'VIRAT_S_010002_03_000236_000261', 'VIRAT_S_010204_03_000606_000632', 'VIRAT_S_010000_07_000827_000860', 'VIRAT_S_010208_03_000201_000232', 'VIRAT_S_010004_03_000239_000277', 'VIRAT_S_010003_07_000608_000636', 'VIRAT_S_010111_04_000718_000760', 'VIRAT_S_010111_08_000920_000954', 'VIRAT_S_010110_00_000000_000021', 'VIRAT_S_010003_02_000165_000202', 'VIRAT_S_010208_05_000591_000631', 'VIRAT_S_010206_02_000414_000439', 'VIRAT_S_010106_02_000656_000692', 'VIRAT_S_010206_05_000797_000823', 'VIRAT_S_010201_08_000705_000739', 'VIRAT_S_010003_10_000901_000934', 'VIRAT_S_010004_00_000064_000100', 'VIRAT_S_010003_06_000526_000560', 'VIRAT_S_010003_08_000739_000778', 'VIRAT_S_010204_10_001372_001395', 'VIRAT_S_010203_09_001010_001036', 'VIRAT_S_010005_07_000535_000584', 'VIRAT_S_010107_02_000282_000312', 'VIRAT_S_010113_05_000776_000805', 'VIRAT_S_010200_06_000702_000744', 'VIRAT_S_010005_08_000647_000693', 'VIRAT_S_010110_03_000712_000739'
    ]
    },
    'DanceTrack': {
        'train': [
            'dancetrack0001', 'dancetrack0052', 'dancetrack0068', 'dancetrack0039', 'dancetrack0086', 'dancetrack0008', 'dancetrack0053', 'dancetrack0045', 'dancetrack0055', 'dancetrack0066', 'dancetrack0062', 'dancetrack0083', 'dancetrack0037', 'dancetrack0023', 'dancetrack0033', 'dancetrack0096', 'dancetrack0029', 'dancetrack0049', 'dancetrack0016', 'dancetrack0044', 'dancetrack0098', 'dancetrack0082', 'dancetrack0015', 'dancetrack0051', 'dancetrack0061', 'dancetrack0087', 'dancetrack0069', 'dancetrack0075', 'dancetrack0020', 'dancetrack0006', 'dancetrack0032', 'dancetrack0012', 'dancetrack0057', 'dancetrack0074', 'dancetrack0024', 'dancetrack0027', 'dancetrack0080', 'dancetrack0002', 'dancetrack0072', 'dancetrack0099'
        ]
    }
}


class opts:
    def __init__(self):  
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--yolo_model',
            type=str,
            default='yolov8m',
            help='YOLO model to use [n, s, m, l, x] and path to .weights file',
        )
        self.parser.add_argument(
            '--visualize',
            default=True,
            action='store_true',
            help='If set, visualizes the video',
        )
        self.parser.add_argument(
            '--eval_mot',
            type=bool,
            default=False,
            help='Uses FasterRCNN detections given by MOT Challenge',
        )
        self.parser.add_argument(
            '--dataset',
            type=str,
            default='MOT17',
            help='MOT17 or MOT20 or KITTI or PersonPath22 or VIRAT-S or DanceTrack',
        )
        self.parser.add_argument(
            '--source',
            default='demo/VIRAT_S_010204_07_000942_000989.mp4',
            type=str,
            help='The path to the video file to be processed'
            )
        self.parser.add_argument(
            '--split',
            type=str,
            default='train',
            help='train or val/test',
        )
        self.parser.add_argument(
            '--tracker_name',
            type=str,
            default='SORT',
            help='LITEDeepSORT or StrongSORT or DeepSORT or SORT',
        )
        self.parser.add_argument(
            '--input_resolution',
            type=int,
            # required=True,
            default=1280,
            help='Resolution for input images (e.g., 1280 for 736x1280)',
        )
        self.parser.add_argument(
            '--min_confidence',
            type=float,
            default=0.25,
            # required=True,
            help='Minimum confidence threshold for detections: default .25',
        )

        self.parser.add_argument(
            '--classes',
            nargs='+',  # '+' means "at least one", '*' for zero or more
            type=int,
            help='For Detection',
            default=[0]  # default list if nothing is provided
        )

        self.parser.add_argument(
            '--appearance_only_matching',  # Corrected typo here
            action='store_true',
            help='If set, skips IOU matching'
        )

        self.parser.add_argument(
            '--BoT',
            action='store_true',
            help='Replacing the original feature extractor with BoT'
        )
        self.parser.add_argument(
            '--ECC',
            action='store_true',
            help='CMC model'
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )
        self.parser.add_argument(
            '--root_dataset',
            default='datasets/'
        )
        self.parser.add_argument(
            '--dir_save',
            default='output/',
            help='e.g, results/MOT17/'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )
        self.parser.add_argument(
            '--max_age',
            type=int,
            default=30
        )
        # add argument for --max_cosine_distance
        self.parser.add_argument(
            '--max_cosine_distance',
            type=float,
            default=0.7
        )
        self.parser.add_argument(
            '--appearance_feature_layer',
            type=str,
            default=None
        )
        # add arg for gpu device
        self.parser.add_argument(
            '--device',
            type=str,
            default='cuda:0'
        )
        self.parser.add_argument(
            '--sequence',
            type=str,
        )
        self.parser.add_argument(
            '--solution',
            type=str,
            default='object_counter',
            help='object_counter, heatmap, etc.'
        )
        self.parser.add_argument(
            '--fps_save',
            default=True,
            action='store_true',
            help='If set, saves FPS results along with the .txt results'
        )

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        if opt.tracker_name == 'StrongSORT':
            # ECC is a complex scenario which offline processing and knowledge of the dataset
            opt.ECC = False
            opt.BoT = True
            opt.NSA = True
            opt.EMA = True
            opt.MC = True
            opt.woC = True
            opt.max_cosine_distance = 0.4
        elif opt.tracker_name.startswith('LITE'):
            opt.woC = True
            opt.ECC = False

        # if opt.max_cosine_distance is none then set it to 0.3
        if opt.max_cosine_distance is None:
            opt.max_cosine_distance = 0.3

        if opt.MC:
            opt.max_cosine_distance += 0.05

        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100

        if opt.ECC:
            path_ECC = f'results/StrongSORT_Git/{opt.dataset}_ECC_{opt.split}.json'
            opt.ecc = json.load(open(path_ECC))
        opt.sequences = data[opt.dataset][opt.split]
        opt.dir_dataset = join(
            opt.root_dataset,
            opt.dataset,
            opt.split
        )
        return opt


opt = opts().parse()
