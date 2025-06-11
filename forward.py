def forward(self, inputs):
    # Always present
    jira_vec = self.bert(**inputs['jira']).pooler_output

    vectors = [jira_vec]  # Start with jira

    # Optional fields — only compute if not just padding
    for key in ['fw_log', 'callstack', 'dmesg', 'driver']:
        input_tensor = inputs[key]
        # Check if the input is NOT just padding (all zeros)
        if input_tensor['input_ids'].sum() > 0:
            vec = self.bert(**input_tensor).pooler_output
            vectors.append(vec)
        else:
            # Append a zero vector if skipped (same dim as pooler_output)
            zero_vec = torch.zeros_like(jira_vec)
            vectors.append(zero_vec)

    # Concatenate all vectors
    combined = torch.cat(vectors, dim=1)
    output = self.classifier(combined)
    return output
