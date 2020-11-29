from model.models.scae import *


def test_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcae = PCAE().to(device)
    temp = torch.randn(5, 1, 28, 28).to(device)
    output = pcae(temp, device)
    for out in output:
        print(out.size())

    ocae = OCAE().to(device)
    output2 = ocae(output[1], output[2], output[3], device)
    for out in output2:
        print(out.size())
