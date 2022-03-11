from script.stage_5_script.Dataset_Loader_Node_Classification import Dataset_Loader
from code_.base_class.result import result
from script.stage_5_script.ModifyBaseClass import Method, Setting, Evaluate_Accuracy


if __name__ == "__main__":
    load_cora = Dataset_Loader(dName='cora')
    load_cite = Dataset_Loader(dName = 'citeseer')
    load_pubm = Dataset_Loader(dName = 'pubmed')
    load_tiny = Dataset_Loader(dName='cora-small')

    load_cora.dataset_source_folder_path = 'data/stage_5_data/cora'
    # load_cite.dataset_source_folder_path = 'data/stage_5_data/citeseer'
    # load_pubm.dataset_source_folder_path = 'data/stage_5_data/pubmed'
    # load_tiny.dataset_source_folder_path = 'data/stage_5_data/cora-small'

    load_method = Method('Graph Method', '')
    load_setting = Setting('Graph Setting', '')
    load_evaluate = Evaluate_Accuracy('accuracy', '')
    load_result = result('save', '')

    print('************ Start ************')
    load_setting.prepare(load_cora, load_method, load_result, load_evaluate)
    # load_setting.prepare(load_cite, load_method, load_result, load_evaluate)
    # load_setting.prepare(load_pubm, load_method, load_result, load_evaluate)
    # load_setting.prepare(load_tiny, load_method, load_result, load_evaluate)
    load_setting.print_setup_summary()

    accuracy = load_setting.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(accuracy))
    print('************ Finish ************')
