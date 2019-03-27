$(function () {
    insertDownloadLinks(files);
});

function insertDownloadLinks(files) {
    $("#download").empty();
    $("#email").empty();

    if (!jQuery.isEmptyObject(files)) {
        if (files['results'])
            $("#download").append("<button id='downloadResults' class='fileSender'>Download results</button>");
        if (files['plot'])
            $("#download").append("<button id='downloadPlot' class='fileSender'>Download plot</button>");
        if (files['results'] || files['plot'])
            $("#email").append("<form id='emailForm'><input type='text' name='emailAddress' placeholder='your@email.com' class='fileSender'><button id='emailSubmit'>Email me</button></form>");
    }
    $("#downloadResults").click(function () {
        download('results');
    });
    $("#downloadPlot").click(function () {
        download('plot');
    });
    $(document).on("submit", "#emailForm", function (event) {
        event.preventDefault();
        $.ajax({
            url: $SCRIPT_ROOT + '/email',
            type: "POST",
            dataType: "JSON",
            data: new FormData(this),
            processData: false,
            contentType: false,
            success: function (data) {
                if (data.status !== 'ok') {
                    alert(data.message);
                }
                else {
                    alert("Results emailed successfully");
                }
            },
            error: function (err) {
                alert("Email failed");
                console.log(err);
            }
        });
    });
}

function download(fileName) {
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/download',
        data: { fileName: fileName },
        success: function (output) {
            if (output.error_message) {
                alert(output.error_message); //even better, add a failure icon with hover
            }
            else {
                var naps_download = new Blob([output], { type: "application / octet - stream" });
                saveAs(naps_download, fileName);
            }
        }
    });
}