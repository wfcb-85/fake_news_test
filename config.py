params = {
'lr':5e-5,
'n_epochs':5,
'batch_size':16,
'transformers_input_type': 'claim_text', #claim text or 'text_from_url' 
'url_text_path': './files/text_from_urls.pb',
'model_name':'custom', #singleTransformer or custom
'claim_author_embedding_dim':32,
'number_items_training_set':1500,
'class_balance':True,
'weight_decay':1e-3,
}
