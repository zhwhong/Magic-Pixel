$(document).ready(function(){
	width = $("#target").outerWidth()
   	$("#imgPlace").css("width",width)
	height = $("#imgPlace").outerHeight()
	console.log(height)
   	$("#preview-pane").css("margin-top", height+10)
   	width = $("#Submit").outerWidth()
   	height = $("#Submit").outerHeight()
   	$("#coords").css("width",width)
   	$("#coords").css("height",height)
   	y = $("#imgPlace").offset().top+$("#imgPlace").outerHeight()
   	x = $("#imgPlace").offset().left+$("#imgPlace").outerWidth()/2
   	t = $("#imgPlace").offset().left
   	console.log(x,y, t)
   	$('#result-pane').css("top",y+10)
   	$('#result-pane').css("left",x+100)
   	width = $("#preview-pane").outerWidth()
   	console.log($("#imgPlace").offset().left)
   	a=  (-1)*$("#imgPlace").offset().left + x -width-100
   	console.log(a)
   	$("#preview-pane").css("left",a)
   	$("#result-pane").css("left", x + 100)

});