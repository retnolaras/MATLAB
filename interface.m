

function varargout = interface(varargin)
% INTERFACE MATLAB code for interface.fig
%      INTERFACE, by itself, creates a new INTERFACE or raises the existing
%      singleton*.
%
%      H = INTERFACE returns the handle to a new INTERFACE or the handle to
%      the existing singleton*.
%
%      INTERFACE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in INTERFACE.M with the given input arguments.
%
%      INTERFACE('Property','Value',...) creates a new INTERFACE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before interface_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to interface_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help interface

% Last Modified by GUIDE v2.5 20-Jun-2016 17:03:10

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @interface_OpeningFcn, ...
                   'gui_OutputFcn',  @interface_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before interface is made visible.
function interface_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to interface (see VARARGIN)

% Choose default command line output for interface
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);



% UIWAIT makes interface wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = interface_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
end

 

% --- Executes on button press in pushbutton2.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

imagen=getimage;
%Imag_seg_ex_2(x);
%% Image segmentation and extraction
%% Read Image
%imagen=imread('test1.jpg');
%% Show image
figure(1)
imshow(imagen);
title('INPUT IMAGE WITH NOISE')
%% Convert to gray scale
%% Convert to binary image

%% Remove all object containing fewer than 30 pixels
%% Show image binary image
figure(2)
imshow(imagen);
title('INPUT IMAGE WITHOUT NOISE')
%% Edge detection
%% Morphology
% * *Image Dilation*
% * *Image Filling*

%rotation

%% Label connected components


%% Measure properties of image regions
%propied=regionprops(L,'BoundingBox');
%hold on
%% Plot Bounding Box
%for n=1:size(propied,1)
   % rectangle('Position',propied(n).BoundingBox,'EdgeColor','g','LineWidth',2)
%end
%hold off
%pause (1)
%% Objects extraction
axes(handles.axes5);
for n=1:Ne
    [r,c] = find(L==n);
    n1=imgn(min(r):max(r),min(c):max(c));
   %imshow(~n1);
    BW2 = bwmorph(n1,'thin',Inf);
    imrotate(BW2,0);
    imshow(~BW2);
    z=imresize(BW2,[50 50]);
    contents = get(handles.popupmenu5,'String'); 
  popupmenu5value = contents{get(handles.popupmenu5,'Value')};
  switch popupmenu5value
      case 'Train using Neural Network'
        z=feature_extractor_NN(z);
      case 'Train using Bagged Decision Tree'
        z=feature_extractor_BDT(z);
      case 'Train using Bagged Neural Network'
        z=feature_extractor_BNN(z);
   end
    load ('D:\training_set\featureout.mat');
    featureout=z;
    %disp(z);
    save ('D:\training_set\featureout.mat','featureout');
    test
    
    pause(0.5);
end
if isempty(re)  %See variable 're' in Fcn 'lines'
        break
    end    
end
clear all
winopen('D:\training_set\output.txt');

close (gcbf)
interface
end

%set(handles.pushbutton9,'Enable','on')


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
end

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
end


% --------------------------------------------------------------------
function Menu_Callback(hObject, eventdata, handles)
% hObject    handle to Menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes on button press in pushbutton6.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% reading the image from the user

% [filename, pathname] = ...
%      uigetfile({'*.jpg';'*.jpeg';'*.png';'*.*'},'Select Image File');
%  I=strcat(pathname,filename); 
% 
%    
%  %  figure(1);
%  %imshow(I);
% axes(handles.axes6);
% imshow(I);
% set(handles.pushbutton13,'Enable','on')
% 
% helpdlg('Image has been Loaded Successfully. Choose an algorithm and train the network  ',...
%         'Load Image');
end
 






% --- Executes during object creation, after setting all properties.
function text8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
end

% --- Executes on button press in pushbutton10.
function pushbutton13_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
contents = get(handles.popupmenu5,'String'); 
  popupmenu5value = contents{get(handles.popupmenu5,'Value')};
  switch popupmenu5value
      case 'Train using Neural Network'
        train
        helpdlg('Network has been trained using Neural Network. Click on "Extract Text" to process the image',...
        'Training Successfull');
      case 'Train using Bagged Decision Tree'
        geo_train
        helpdlg('Network has been trained using Bagged Decision Tree. Click on "Extract Text" to process the image',...
        'Training Successfull');
    case 'Train using Bagged Neural Network'
        geo_train
        helpdlg('Network has been trained using Bagged Neural Network. Click on "Extract Text" to process the image',...
        'Training Successfull');
  end
    set(handles.pushbutton11,'Enable','on')
end
end


% --------------------------------------------------------------------
% function Help_Callback(hObject, eventdata, handles)
% % hObject    handle to Help (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% open ReadMe.pdf


% --------------------------------------------------------------------
% function About_us_Callback(hObject, eventdata, handles)
% % hObject    handle to About_us (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% open aboutus.fig

