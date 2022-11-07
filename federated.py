import time
import torch
import math
from core.domain import create_domains, create_multi_domain
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.DQNScheduler import DQNScheduler
from analyzer.results_analyzer import compute_avg_task_process_time_by_name
from utils.load_data import load_machines_from_file, load_task_batches_from_file, sample_tasks_from_file, \
    load_tasks_from_file, sample_task_batches_from_file
from utils.file_check import check_and_build_dir
from utils.state_representation import get_machine_kind_list
import globals.global_var as glo


def create_clients(clients_num):
    client_list = []
    for client_id in range(clients_num):
        """Perform inter-domain task scheduling
        """
        # 1. create multi-domain system
        multi_domain_system_location = "北京市"
        multi_domain = create_multi_domain(client_id, multi_domain_system_location)
    
        # 2. create domains
        domain_num = 5
        location_list = ["北京市", "上海市", "莫斯科", "新加坡市", "吉隆坡"]
        domain_list = create_domains(location_list)
    
        # 3. add machines to domain
        machine_file_path = glo.machine_file_path
        machine_list = load_machines_from_file(machine_file_path)
        machine_num_per = len(machine_list) // domain_num
        for domain_id in range(domain_num):
            for i in range(machine_num_per):
                machine = machine_list[i + domain_id*machine_num_per]
                machine.set_location(domain_list[domain_id].longitude, domain_list[domain_id].latitude)
                domain_list[domain_id].add_machine(machine)
    
        # 4. clustering machines in each domain
        cluster_num = 3
        for domain in domain_list:
            domain.clustering_machines(cluster_num)
    
        # 5. add domain to multi-domain system
        for domain in domain_list:
            multi_domain.add_domain(domain)
            
        # 7. set scheduler for multi-domain system
        machine_num = len(machine_list)
        # scheduler = RoundRobinScheduler(machine_num)
        machine_kind_num_list, machine_kind_idx_range_list = get_machine_kind_list(machine_list)
        
        # 第一种方式，改epsilon_decay
        # target = 0.9998
        # target = 1.05
        # init = 0.975
        # diff = target - init
        # epochs = 10
        # epsilon_inc = diff / epochs
        # epsilon_dec = init + epsilon_inc * epoch
        # epsilon_dec = 0.998
        
        # 第二种方式，改action average
        # init = 1
        # target = 0.3
        # diff = init - target
        # epochs = 10
        # prob_decay = diff / epochs
        # prob = init - prob_decay * epoch
        
        # 第三种方式，改balance_prob
        # target = 0.3
        # init = 1.0
        # diff = target - init
        # epochs = 100
        # bprob_step = diff / epochs
        # balance_prob = init - bprob_step * epoch
        
        init_epsilon = 0.95
        epsilon_dec = 0.999
        prob = 0.5
        balance_prob = 0.3
        # balance_prob = 0.1
    
        scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, machine_kind_num_list,
                                 machine_kind_idx_range_list, is_federated=True, init_epsilon=init_epsilon, epsilon_decay=epsilon_dec, prob=prob, balance_prob=balance_prob)
        scheduler_name = scheduler.__class__.__name__
        glo.current_scheduler = scheduler_name
        multi_domain.set_scheduler(scheduler)
        
        client_list.append(multi_domain)
    return client_list
        

def client_train(multi_domain, client_id):
    # 6. load tasks
    # task_file_path = f"dataset/GoCJ/client/GoCJ_Dataset_2000_client_{client_id}.txt"
    task_file_path = f"dataset/Alibaba/client/Alibaba-Cluster-trace-100000-client-{client_id}.txt"
    # task_batch_list = load_task_batches_from_file(task_file_path, delimiter='\t')
    task_batch_list = sample_task_batches_from_file(task_file_path, batch_num=1, delimiter='\t')
    
    task_batch_num = len(task_batch_list)
    multi_domain.scheduler.set_max_step(task_batch_num)

    # 8. commit tasks to multi-domain system, training
    glo.is_test = False
    for batch in task_batch_list:
        multi_domain.commit_tasks(batch)
    # multi_domain.commit_tasks(task_batch_list)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. show statistics
    # compute_avg_task_process_time_by_name(scheduler_name)


def federated_test(epoch):
    """Perform inter-domain task scheduling
    """
    # 1. create multi-domain system
    multi_domain_system_location = "北京市"
    client_id = 10000   # federated server
    multi_domain = create_multi_domain(client_id, multi_domain_system_location)

    # 2. create domains
    domain_num = 5
    location_list = ["北京市", "上海市", "莫斯科", "新加坡市", "吉隆坡"]
    domain_list = create_domains(location_list)

    # 3. add machines to domain
    machine_file_path = glo.machine_file_path
    machine_list = load_machines_from_file(machine_file_path)
    machine_num_per = len(machine_list) // domain_num
    for domain_id in range(domain_num):
        for i in range(machine_num_per):
            machine = machine_list[i + domain_id*machine_num_per]
            machine.set_location(domain_list[domain_id].longitude, domain_list[domain_id].latitude)
            domain_list[domain_id].add_machine(machine)

    # 4. clustering machines in each domain
    cluster_num = 3
    for domain in domain_list:
        domain.clustering_machines(cluster_num)

    # 5. add domain to multi-domain system
    for domain in domain_list:
        multi_domain.add_domain(domain)

    # 6. load tasks
    # task_file_path = f"dataset/GoCJ/GoCJ_Dataset_5000records_50concurrency_test.txt"
    task_file_path = f"dataset/Alibaba/Alibaba-Cluster-trace-500000-test.txt"
    # task_file_path = f"dataset/Alibaba/Alibaba-Cluster-trace-5000-test.txt"
    tasks_for_test = load_task_batches_from_file(task_file_path, delimiter='\t')

    # 7. set scheduler for multi-domain system
    machine_num = len(machine_list)
    task_batch_num = len(tasks_for_test)
    # scheduler = RoundRobinScheduler(machine_num)
    machine_kind_num_list, machine_kind_idx_range_list = get_machine_kind_list(machine_list)
    
    # 第一种方式，改epsilon_decay
    # target = 0.9998
    # target = 1.05
    # init = 0.975
    # diff = target - init
    # epochs = 10
    # epsilon_inc = diff / epochs
    # epsilon_dec = init + epsilon_inc * epoch
    # epsilon_dec = 0.998
    
    # 第二种方式，改action average
    # init = 1
    # target = 0.3
    # diff = init - target
    # epochs = 10
    # prob_decay = diff / epochs
    # prob = init - prob_decay * epoch
    
    # 第三种方式，改balance_prob
    # target = 0.6
    # init = 0.7
    # diff = init - target
    # epochs = 50
    # bprob_step = diff / epochs
    # balance_prob = init - bprob_step * (epoch - 100)
    # if epoch > 150:
    #     balance_prob = 0.6
    
    init_epsilon = 0.1
    epsilon_dec = 0.998
    prob = 0.5
    balance_prob = 0.3
    
    scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, machine_kind_num_list,
                             machine_kind_idx_range_list, is_federated=True, init_epsilon=init_epsilon, epsilon_decay=epsilon_dec, prob=prob, balance_prob=balance_prob)
    scheduler_name = scheduler.__class__.__name__
    glo.current_scheduler = scheduler_name
    multi_domain.set_scheduler(scheduler)
    
    scheduler.set_max_step(task_batch_num)

    # 8. commit tasks to multi-domain system, training
    glo.is_test = True
    for batch in tasks_for_test:
        multi_domain.commit_tasks(batch)
    # multi_domain.commit_tasks(tasks_for_test)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. show statistics
    # compute_avg_task_process_time_by_name(scheduler_name)


def test_federated():
    # Initialization
    start_time = time.time()
    n_clients = 10
    federated_rounds = 100
    glo.is_federated = True
    glo.is_test = False
    glo.is_print_log = False
    init_federated_model()
    client_list = create_clients(n_clients)

    # federated main
    print("federated learning start...")
    for epoch in range(0, federated_rounds):
        glo.federated_round = epoch
        print(f"Round {epoch}")
        for client_id in range(n_clients):
            print(f"Client-{client_id}:")
            client_train(client_list[client_id], client_id)
        fed_avg(n_clients)
        glo.is_test = True
        federated_test(epoch)
        # for i in range(10):     # test 10 times and get average
        #     federated_test(epoch)
        glo.is_test = False

    print("federated learning finished.")
    end_time = time.time()
    print("Time used: %.2f s" % (end_time - start_time))


def fed_avg(clients_num):
    model_path_list = []
    for i in range(clients_num):
        model_path_list.append(f"save/client-{i}/eval.pth")

    # load client weights
    clients_weights_sum = None
    for model_path in model_path_list:
        cur_parameters = torch.load(model_path)
        if clients_weights_sum is None:
            clients_weights_sum = {}
            for key, var in cur_parameters.items():
                clients_weights_sum[key] = var.clone()
        else:
            for var in cur_parameters:
                clients_weights_sum[var] = clients_weights_sum[var] + cur_parameters[var]

    # fed_avg
    global_weights = {}
    for var in clients_weights_sum:
        global_weights[var] = (clients_weights_sum[var] / clients_num)
    global_model_path = f"save/global/global.pth"
    torch.save(global_weights, global_model_path)


def init_federated_model():
    machine_num = 20
    scheduler = DQNScheduler(multidomain_id=1, machine_num=machine_num,
                             machine_kind_num_list=[], machine_kind_idx_range_list=[],
                             is_federated=False)
    global_model_dir = "save/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"save/global/global.pth"
    scheduler.DRL.save_initial_model(global_model_path)


if __name__ == "__main__":
    test_federated()
    # init_federated_model()
