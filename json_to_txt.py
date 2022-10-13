import json
import os
import yaml
import requests
from PIL import Image
from tqdm import tqdm
import shutil

# PATH to the train and validation labels in JSON format
#train_json_path = "../../dataset/labels/train.json"
#valid_json_path = "../../dataset/labels/validation.json"
train_json_path = "/home/ashimag/wii_data_species_2022/labels/wii_aite_2022_0.8.json"
# train_json_path = "/home/ashimag/wii_data_tiger_2014/wii_data_tiger_project_2014_new.json"
# train_json_path = "/home/ashimag/WII-CameraTrap-2018/Saket_IIITD/CaTRAT_B_W/wii_CaTRAT_B_W_2018.json"
# train_json_path = "/home/ashimag/WII-CameraTrap-2018/Saket_IIITD/CaTRAT_Species_Training_Pictures_/wii_CaTRAT_Species_Training_Pictures_2018.json"

#82 classes
# classes= ['mani_cras-Manis crassicaudata', 'maca_munz-Macaca munzala', 'maca_radi-Macaca radiata', 'athe_macr', 'vulp_beng', 'lept_java-Leptoptilos javanicus',
#  'trac_pile-Trachypithecus pileatus', 'hyst_brac-Hystrix brachyura', 'nilg_hylo-Nilgiritragus hylocrius', 'prio_vive-Prionailurus viverrinus',
#   'neof_nebu-Neofelis nebulosa', 'melu_ursi', 'vehi_vehi', 'hyae_hyae-Hyaena hyaena', 'maca_mula-Macaca mulatta', 'fran_pond-Francolinus pondicerianus',
#    'munt_munt-Muntiacus muntjak', 'feli_sylv-Felis sylvestris', 'maca_sile-Macaca silenus', 'vive_zibe-Viverra zibetha', 'rusa_unic-Rusa unicolor',
#     'lepu_nigr-Lepus nigricollis', 'vive_indi-Viverricula indica', 'pavo_cris', 'anti_cerv', 'gall_lunu-Galloperdix lunulata', 'cato_temm-Catopuma temminckii',
#      'sus__scro-Sus scrofa', 'cani_aure-Canis aureus', 'para_herm-Paradoxurus hermaphroditus', 'axis_axis', 'catt_kill', 'goat_sheep', 'vara_beng-Varanus bengalensis',
#       'para-jerd-Paradoxurus jerdoni', 'mart_gwat-Martes gwatkinsii', 'homo_sapi', 'semn_john+Semnopithecus johnii', 'herp_edwa-Herpestes edwardsii', 'bos__fron',
#        'herp_vitt-Herpestes vitticollis', 'arct_coll', 'dome_cats-Domestic cat', 'bos__indi', 'mell_cape-Mellivora capensis', 'ursu_thib-Ursus thibetanus',
#         'semn_ente-Semnopithecus entellus', 'prio_rubi-Prionailurus rubiginosus', 'dome_dogs-Domestic dog', 'cani_lupu-Canis lupus', 'gall_sonn-Gallus sonneratii',
#          'gaze_benn-Gazella bennettii', 'bose_trag-Boselaphus tragocamelus', 'budo_taxi-Budorcas taxicolor', 'bos__gaur', 'catt_catt-Cattle', 'blan_blan',
#           'cuon_alpi-Cuon alpinus', 'capr_thar-Capricornis thar', 'equu_caba-Equus caballus', 'herp_fusc-Herpestes fuscus', 'trac_john-Trachypithecus johnii',
#            'vara_salv-Varanus salvator', 'gall_gall-Gallus gallus', 'naem_gora-Naemorhedus goral', 'herp_urva-Herpestes urva', 'hyst_indi-Hystrix indica',
#             'herp_smit-Herpestes smithii', 'bird_bird', 'tetr_quad-Tetracerus quadricornis', 'feli_chau-Felis chaus', 'maca_arct-Macaca arctoides',
#              'lutr_pers-Lutrogale perspicillata', 'mosc_indi-Moschiola indica', 'pant_tigr', 'pant_pard-Panthera pardus', 'mart_flav-Martes flavigula',
#               'pagu_larv-Paguma larvata-Masked Palm Civet', 'prio_beng-Prionailurus bengalensis', 'gall_spad-Galloperdix spadicea', 'elep_maxi-Elephas maximus',
#                'axis_porc']   # class names
# 98 classes
classes= ['mani_cras-Manis crassicaudata', 'maca_munz-Macaca munzala', 'maca_radi-Macaca radiata', 'athe_macr', 'vulp_beng', 'lept_java-Leptoptilos javanicus',
 'trac_pile-Trachypithecus pileatus', 'hyst_brac-Hystrix brachyura', 'nilg_hylo-Nilgiritragus hylocrius', 'prio_vive-Prionailurus viverrinus',
  'neof_nebu-Neofelis nebulosa', 'melu_ursi', 'vehi_vehi', 'hyae_hyae-Hyaena hyaena', 'maca_mula-Macaca mulatta', 'fran_pond-Francolinus pondicerianus',
   'munt_munt-Muntiacus muntjak', 'feli_sylv-Felis sylvestris', 'maca_sile-Macaca silenus', 'vive_zibe-Viverra zibetha', 'rusa_unic-Rusa unicolor',
    'lepu_nigr-Lepus nigricollis', 'vive_indi-Viverricula indica', 'pavo_cris', 'anti_cerv', 'gall_lunu-Galloperdix lunulata', 'cato_temm-Catopuma temminckii',
     'sus__scro-Sus scrofa', 'cani_aure-Canis aureus', 'para_herm-Paradoxurus hermaphroditus', 'axis_axis', 'catt_kill', 'goat_sheep', 'vara_beng-Varanus bengalensis',
      'para-jerd-Paradoxurus jerdoni', 'mart_gwat-Martes gwatkinsii', 'homo_sapi', 'semn_john+Semnopithecus johnii', 'herp_edwa-Herpestes edwardsii', 'bos__fron',
       'herp_vitt-Herpestes vitticollis', 'arct_coll', 'dome_cats-Domestic cat', 'bos__indi', 'mell_cape-Mellivora capensis', 'ursu_thib-Ursus thibetanus',
        'semn_ente-Semnopithecus entellus', 'prio_rubi-Prionailurus rubiginosus', 'dome_dogs-Domestic dog', 'cani_lupu-Canis lupus', 'gall_sonn-Gallus sonneratii',
         'gaze_benn-Gazella bennettii', 'bose_trag-Boselaphus tragocamelus', 'budo_taxi-Budorcas taxicolor', 'bos__gaur', 'catt_catt-Cattle', 'blan_blan',
          'cuon_alpi-Cuon alpinus', 'capr_thar-Capricornis thar', 'equu_caba-Equus caballus', 'herp_fusc-Herpestes fuscus', 'trac_john-Trachypithecus johnii',
           'vara_salv-Varanus salvator', 'gall_gall-Gallus gallus', 'naem_gora-Naemorhedus goral', 'herp_urva-Herpestes urva', 'hyst_indi-Hystrix indica',
            'herp_smit-Herpestes smithii', 'bird_bird', 'tetr_quad-Tetracerus quadricornis', 'feli_chau-Felis chaus', 'maca_arct-Macaca arctoides',
             'lutr_pers-Lutrogale perspicillata', 'mosc_indi-Moschiola indica', 'pant_tigr', 'pant_pard-Panthera pardus', 'mart_flav-Martes flavigula',
              'pagu_larv-Paguma larvata-Masked Palm Civet', 'prio_beng-Prionailurus bengalensis', 'gall_spad-Galloperdix spadicea', 'elep_maxi-Elephas maximus',
               'axis_porc', 'anat_elli', 'bats_bats', 'call_pyge-Callosciurus pygerythrus',
'came_came-Camel', 'capr_hisp-Caprolagus hispidus', 'funa_palm-Funambulus palmarum', 'hela_mala-Helarctos malayanus', 'lutr_lutr-Lutra lutra', 'maca_assa-Macaca assamensis', 'maca_leon-Macaca leonina', 'maca_maca-Macaque', 
'melo_pers', 'pard_marm-Pardofelis marmorata', 'prio_pard-Prionodon pardicolor', 'tree_shre', 'vulp_vulp']   # class names

# class xcentre ycentre width height  <-- normalised values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(classes)
word_to_int=le.transform(classes)
res = dict(zip(classes, word_to_int))



#import pdb; pdb.set_trace()

def bbox_to_yolo(bbox):
    if len(bbox) == 0:
        return "0.5 0.5 0.9999 0.9999"
    x_centre = str(bbox[0] + (bbox[2] / 2))
    y_centre = str(bbox[1] + (bbox[3] / 2 ))
    wi = str(bbox[2])
    hi = str(bbox[3])
    return x_centre + " " + y_centre + " " + wi + " " + hi

no_bbox=[]
excluded=[]
if __name__ == "__main__":
    confirmation = "x"
    # confidence=0.5
    blan_train= "/home/ashimag/wii_data_species_2022/images/train/blan_blan"
    blan_val="/home/ashimag/wii_data_species_2022/images/validation/blan_blan"
    blan_test="/home/ashimag/wii_data_species_2022/images/test/blan_blan"
    blan_image_train=os.listdir(blan_train)
    blan_image_val=os.listdir(blan_val)
    blan_image_test=os.listdir(blan_test)
    blan_image = blan_image_train+blan_image_val+blan_image_test
    print(f'No of blank images:{len(blan_image)}')
    print(f'Sample image:{blan_image[0]}')
    
    # path = "../../dataset/labels/train/"  # update the path where the training labels txt files will be saved
    path= "/home/ashimag/wii_data_species_2022/labels/wii_all_labels_round2_revised/"
    # path = "/home/ashimag/wii_data_tiger_2014/all_labels/"
    # path = "/home/ashimag/WII-CameraTrap-2018/Saket_IIITD/CaTRAT_B_W/Originals_CaTRAT_labels/"
    # path ="/home/ashimag/WII-CameraTrap-2018/Saket_IIITD/CaTRAT_Species_Training_Pictures_/labels/"
    f = open(train_json_path)
       

    if os.path.isdir(path) and confirmation != "y":
        confirmation = input("Path '%s" %path + "' already exists. Overwrite? (y/n)")
        
    elif not os.path.isdir(path):
        os.mkdir(path)

    print("Transforming labels!")


    check_set = set()
    training_data = json.load(f)
    blan_blan=0
    no_bbox=0
    low_conf=0
    rest=0
    total=0
    empty=[]
    for i, ann in enumerate(training_data["images"]):
        
        if ann['file'].split('/')[-1] in blan_image:
            
            print('Blank images')
            category_name = str(ann['file'].split('/')[-2])
            if not os.path.exists(os.path.join(path,category_name )) and (category_name in res):
                os.mkdir(os.path.join(path,category_name))
            # category_id = res[category_name]
            # name = str(ann['file'].split('/')[-1].split('.')[0])
            # content = str(category_id) + " " + "0.0 0.0 0.0 0.0"
            # file_name = os.path.join(path, category_name, f'{name}.txt')
            # # import pdb; pdb.set_trace()
            # file = open(file_name, "a")
            # file.write("\n")
            # file.write(content)
            # file.close()
            # blan_blan+=1
            # total+=1
            continue

        elif "detections" not in ann:
            print('No detection in json')
            #move train image to test folder
            destination= "/home/ashimag/wii_data_species_2022/empty"
            if not os.path.exists(destination):
                    os.mkdir(destination)
            # shutil.move( destination,ann['file'],)
            empty.append(ann['file'])
            total+=1
            continue
        else:
            # total+=1
            if (len(ann["detections"])==0):
                total+=1
                no_bbox+=1
                print('Empty Bbox')
            #     # excluded.append(ann['file'])
            #     no_bbox.append(ann['file'])
                im_path = ann["file"]
                category_name = str(im_path.split('/')[-2])
                # dir_new=f"/home/ashimag/wii_data_species_2022/labels/test_70_final/{category_name}/"
                # dir_new= os.path.join(path, category_name)
                # if not os.path.exists(dir_new) and (category_name in res):
                #     os.mkdir(dir_new)
                destination= "/home/ashimag/wii_data_species_2022/empty"
                if not os.path.exists(destination):
                    os.mkdir(destination)
                empty.append(ann['file'])
                # shutil.move( destination,ann['file'])
                # if category_name not in res:
                #     continue
                # category_id = res[category_name]
                # name = str(im_path.split('/')[-1].split('.')[0])
                # file_name = os.path.join(dir_new, f'{name}.txt')
                # # import pdb; pdb.set_trace()
                # content = str(category_id) + " " + "0.5 0.5 0.9999 0.9999"
                # file = open(file_name, "a")
                # file.write("\n")
                # file.write(content)
                # file.close()
                continue
            else:   
                total+=1
                list_bbox= ann["detections"]
                category_name = str(ann['file'].split('/')[-2])
                
                dir_new=os.path.join(path,category_name)
                if not os.path.exists(dir_new) and (category_name in res):
                    os.mkdir(dir_new)
                if category_name== 'blan_blan':
                    continue
                if category_name not in res:
                    continue
                category_id = res[category_name]
                name = str(ann['file'].split('/')[-1].split('.')[0])
                file_name= os.path.join(path,category_name, f'{name}.txt' )
                # import pdb; pdb.set_trace()
                c=0
                for item in list_bbox:
                    
                    if (float(item['conf']) >=0.4) and (int(item['category'])==1): 
                        if c==0:
                            rest+=1 
                        c=1
                           
                        bbox = item['bbox']  # "bbox": [x,y,width,height]
                        content = str(category_id) + " " + bbox_to_yolo(bbox)
                    # # print(content)
                        if name in check_set:
                            # print('writing in same txt')
                            print(f'same txt :{name}')
                            file = open(file_name, "a")
                            file.write("\n")
                            file.write(content)
                            file.close()
                        else:
                            check_set.add(name)
                            file = open(file_name, "w")
                            file.write(content)
                            file.close()

                    elif (float(item['conf']) > 0.5) and (int(item['category']) == 2 or int(item['category']) == 3): 
                        if c==0:
                            rest+=1 
                        c=1
                        bbox = item['bbox'] 
                        if int(item['category'])== 2:
                            category_id= res['homo_sapi']
                            # import pdb; pdb.set_trace()
                            print(f'Id is {category_id}_human detected')
                        elif int(item['category'])== 3:
                            category_id= res['vehi_vehi']
                            print(f'Id is {category_id}_vehicle')

                        #  "bbox": [x,y,width,height]
                        imag = training_data['images'][i]
                        content = str(category_id) + " " + bbox_to_yolo(bbox)
                        if name in check_set:
                            print(f'same txt :{name}')
                            file = open(file_name, "a")
                            file.write("\n")
                            file.write(content)
                            file.close()
                        else:
                            check_set.add(name)
                            file = open(file_name, "w")
                            file.write(content)
                            file.close()
                
                    # break
                    
                if c==0:   
                    low_conf+=1
                    print('Low Conf')
                    
                    dir_new=f"/home/ashimag/wii_data_species_2022/labels/low_conf_70/{category_name}"
                    # dir_new= os.path.join(path, category_name)
                    if not os.path.exists(dir_new) and (category_name in res):
                        os.mkdir(dir_new)
                    if category_name not in res:
                        continue
                    if category_name== 'blan_blan':
                        continue
                    category_id = res[category_name]
                    name = str(ann['file'].split('/')[-1].split('.')[0])
                    file_name= os.path.join(dir_new,f'{name}.txt')
                    # import pdb; pdb.set_trace()
                    for item in list_bbox:

                            
                        bbox = item['bbox']  # "bbox": [x,y,width,height]
                        content = str(category_id) + " " + bbox_to_yolo(bbox)
                    # print(content)
                        if name in check_set:
                            # print('writing in same txt')
                            print(f'same txt :{name}')
                            file = open(file_name, "a")
                            file.write("\n")
                            file.write(content)
                            file.close()
                        else:
                            check_set.add(name)
                            file = open(file_name, "w")
                            file.write(content)
                            file.close()
        #             # break
    print(f'Total no of images:{total}')
    print(f'no bounding box except blank blank count: {no_bbox}')
    print(f'blank blank images count: {blan_blan}')
    print(f'Low confidence images count: {low_conf}')
    print(f'Remaining images:{rest}')


    # print(f'final total excluded images{total}')
    # print(len(no_bbox))
    # print(len(excluded))

    with open(f'/home/ashimag/wii_data_species_2022/labels/empty.txt', 'w') as fp:
        fp.write('\n'.join(empty))

    # with open(f'/home/ashimag/wii_data_species_2022/labels/excluded.txt', 'w') as fp:
    #     fp.write('\n'.join(excluded))
