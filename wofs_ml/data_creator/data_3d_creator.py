from data_3d_helper import *

#This data is used to grab all the example and label data
#from the 3d unet and create data the is usable for the 3d unet

#create parser
parser = create_parser()
args = parser.parse_args()

run_num = args.run_num


if run_num == 0:
    get_2017_2018_data()
elif run_num == 1:
    get_2019_data()
elif run_num == 2:
    get_2020_2021_data()
elif run_num == 3:
    merge_examples()





