import torch
from src.models.DECIFRA_MS import DECIFRA_MS

class DECIFRA_IG(DECIFRA_MS):
    """
    Experimental subclass of DECIFRA_MS for computing Isolated Integrated Gradients.
    It overrides the forward method to allow injecting `forced_matrices` and `forced_h_pre_btp`.
    """
    def forward(self, x, forced_matrices=None, forced_h_pre_btp=None): 
        B, T, C = x.shape  # [batch_size, time_length, input_size==self.input_size]
        orig_x = x

        E = self.model_cfg.rnn.input_embedding_size
        H = self.model_cfg.rnn.hidden_size

        # 1) Embed all input signals
        x_emb = self.embed_signals(x)  # [B, T, C, E]

        # 2) Run recurrent loop
        h = torch.zeros(B, C, H, device=x.device)

        transfer_matrices = []
        hidden_states = []
        hidden_states_pre_btp = []
        for t in range(T):
            # Process one time step

            # prepare GRU input
            x_input = x_emb[:, t, :, :].reshape(B*C, 1, E) # [B*C, 1, E] = [effective_GRU_batch, seq_len=1, GRU_input_dim]
            h_input = h.reshape(1, B*C, H) # [1, B*C, hidden_dim] = [bidirectional_GRU*GRU_num_layers = 1, effective_GRU_batch, hidden_size]

            if forced_h_pre_btp is not None:
                # Bypass GRU, use true pre-BTP hidden states to sever recurrence
                new_h = forced_h_pre_btp[:, t]
            else:
                if self.model_cfg.single_GRU:
                    _, new_h = self.gru(x_input, h_input) # output h shape is the same: (1, GRU_batch, hidden_size)

            h = new_h.reshape(B, C, H) # (B, C, H)
            
            # Save true pre-BTP states during the clean pass
            hidden_states_pre_btp.append(h.clone())

            if forced_matrices is not None:
                mixing_matrix = forced_matrices[:, t, :, :]
                h = torch.bmm(mixing_matrix, h)
            else:
                # Derive transition matrix and mix hidden states
                h, mixing_matrix = self.BTP(h)

            # save outputs
            hidden_states.append(h)
            transfer_matrices.append(mixing_matrix)

            if torch.any(torch.isnan(h)): # for debugging, h will have invalid values if model diverges
                raise Exception(f"h has nans at time point {t}")

        # Stack the transition matrices, predict the next input
        transfer_matrices = torch.stack(transfer_matrices, dim=1)  # (batch_size, seq_len, input_size, input_size)
        hidden_states = torch.stack(hidden_states, dim=1)[:, :-1, :, :] # brain latent states starting with time 0, [batch_size; time_length-1; input_size, hidden_dim]
        hidden_states_pre_btp = torch.stack(hidden_states_pre_btp, dim=1)
        predicted = self.predictor(hidden_states).squeeze() # predictions of x starting with time 1, [batch_size; time_length-1; input_size]
        
        if self.pretraining:
            # pretrain on the forecasting task
            return None, {
                "matrices": transfer_matrices,
                "predicted": predicted,
                "originals": orig_x,
                "h_pre_btp": hidden_states_pre_btp
            }
        
        clf_input = transfer_matrices.reshape(B, T, -1) # [batch_size; time_length; input_size * input_size]
        time_logits = self.clf(clf_input) # [batch_size; time_length, n_classes]
        logits = torch.mean(time_logits, dim=1) # mean over time, [batch_size; n_classes]

        loss_load = {
            "logits": logits,
            "matrices": transfer_matrices,
            "time_logits": time_logits,
            "predicted": predicted,
            "originals": orig_x,
            "h_pre_btp": hidden_states_pre_btp
        }
        return logits, loss_load
