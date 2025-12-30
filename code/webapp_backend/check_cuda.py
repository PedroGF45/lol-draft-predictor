import torch

print("cuda_available=", torch.cuda.is_available())
print("cuda_version=", torch.version.cuda)
print("device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"device[{i}]= {props.name}, capability={props.major}.{props.minor}, memory={props.total_memory/1024**3:.1f} GiB"
        )
