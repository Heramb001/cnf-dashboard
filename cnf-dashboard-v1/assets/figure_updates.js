//--- reset all the data on dashboard
$(document).on('click','#reset-button', function(){
    var input_graph = document.getElementById('g1');
    var output_graph = document.getElementById('output-graph-g2');
    in_data_update={};
    in_layout_update={xaxis:{title:{text:'X'}},yaxis:{title:{text:'Q2'}}};
    out_data_update={};
    out_layout_update={xaxis:{title:{text:'Parameters'}},yaxis:{title:{text:'Normalized values'}}};
    Plotly.update(input_graph, in_data_update, in_layout_update);
    Plotly.update(output_graph, out_data_update, out_layout_update);
});