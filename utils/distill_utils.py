import copy

import torch
import torch.nn.functional as F


def warmup_teacher_head(model, train_loader, task_name, device, epochs=5, lr=1e-3, weight_decay=0.0, log_fn=print):
    teacher = copy.deepcopy(model)
    teacher.to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    head = teacher.last[task_name]
    for parameter in head.parameters():
        parameter.requires_grad = True

    teacher.eval()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    metrics = []
    for epoch in range(int(epochs)):
        loss_sum = 0.0
        correct = 0
        total = 0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = teacher(inputs)[task_name]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(inputs.size(0))
        avg_loss = loss_sum / max(1, total)
        avg_acc = 100.0 * correct / max(1, total)
        metrics.append({'epoch': epoch, 'loss': avg_loss, 'acc': avg_acc})
        log_fn('Teacher warmup epoch {} loss {:.4f} acc {:.2f}'.format(epoch, avg_loss, avg_acc))

    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    return teacher, metrics


def compute_intra_task_distill_loss(student_logits, teacher_logits, tau_distill=2.0):
    temperature = max(float(tau_distill), 1e-6)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
