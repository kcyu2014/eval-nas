
def nasbench_model_forward(model, input, target, criterion):
    logits, aux_logits = model(input)
    loss = criterion(logits, target)
    if aux_logits is not None:
        aux_loss = criterion(aux_logits, target)
        loss += 0.4 * aux_loss
    return loss, logits, aux_logits