let file=document.getElementById('upload');
let button=document.getElementsByTagName('button');
let progress = document.querySelector('progress');

let p_i=document.querySelector('.progress-indicator');
let load=0;
let proces="";

file.oninput=()=>
{
    let filename=file.files[0].name;
    let extension=filename.split('.').pop();
    let filesize=file.files[0].size;
    if(filesize<=1000000)
    {
        filesize=(filesize/1000).toFixed(2)+'kb';
    }
    if(filesize==1000000||filesize<=1000000000)
    {
        filesize=(filesize/1000000).toFixed(2)+'mb';
    }
    if(filesize==1000000000||filesize<=1000000000000)
    {
        filesize=(filesize/1000000000).toFixed(2)+'gb';
    }
    document.querySelector('label').innerText=filename;
    document.querySelector('.ex').innerText=extension;
    document.querySelector('.size').innerText=filesize;
    getFile(filename);
}
let upload=()=>{
    if(load>=100)
    {
        clearInterval(proces);
        p_i.innerHTML='100%'+ ' ' + 'upload completed';
        button[0].classList.remove('active');
    }
    else{
        load++;
        progress.value=load;
        p_i.innerHTML=load+'%'+' '+'upload';
        button[1].onclick=e=>
        {
            e.preventDefault();
            clearInterval(proces);
            document.querySelector('.pr').style.display="none";
            button[1].style.visibility='hidden';
            button[0].classList.remove('active');
        }
    }
}
function getFile(filename)
{
    if(filename)
    {
        document.querySelector('.pr').style.display="block";
        load=0;
        progress.value=0;
        p_i.innerText='';
        button[0].onclick=e=>
        {
            e.preventDefault();
            button[0].classList.add('active');
            button[1].style.visibility='visible';
            proces=setInterval(upload,100);
        }
    }
}

