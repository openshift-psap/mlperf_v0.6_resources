import subprocess
import xml.etree.ElementTree as ET

from logger.logger import LOGGER

#TODO: print CUDA version, container version etc

def log_hardware():
    # TODO: asserts - what if you cannot launch those commands?
    # number of CPU threads
    cpu_info_command = 'cat /proc/cpuinfo'
    cpu_info = subprocess.run(cpu_info_command.split(), stdout=subprocess.PIPE).stdout.split()
    cpu_num_index = len(cpu_info) - cpu_info[::-1].index(b'processor') + 1
    cpu_num = int(cpu_info[cpu_num_index]) + 1

    # CPU name
    cpu_name_begin_index = cpu_info.index(b'name')
    cpu_name_end_index = cpu_info.index(b'stepping')
    cpu_name = b' '.join(cpu_info[cpu_name_begin_index + 2:cpu_name_end_index]).decode('utf-8')

    LOGGER.log(key='cpu_info', value={"num": cpu_num, "name": cpu_name}, stack_offset=1)

    # RAM memory
    ram_info_command = 'free -m -h'
    ram_info = subprocess.run(ram_info_command.split(), stdout=subprocess.PIPE).stdout.split()
    ram_index = ram_info.index(b'Mem:') + 1
    ram = ram_info[ram_index].decode('utf-8')

    LOGGER.log(key='mem_info', value={"ram": ram}, stack_offset=1)

    # GPU
    nvidia_smi_command = 'nvidia-smi -q -x'
    nvidia_smi_output = subprocess.run(nvidia_smi_command.split(), stdout=subprocess.PIPE).stdout
    nvidia_smi = ET.fromstring(nvidia_smi_output)
    gpus = nvidia_smi.findall('gpu')
    ver = nvidia_smi.findall('driver_version')

    LOGGER.log(key="gpu_info",
                 stack_offset=1,
                 value={
                      "driver_version": ver[0].text,
                      "num": len(gpus),
                      "name": [g.find('product_name').text for g in gpus],
                      "mem": [g.find('fb_memory_usage').find('total').text for g in gpus]})

def log_args(args):
    LOGGER.log(key='args', value=vars(args), stack_offset=1)
