import imageio as iio

def write_video(rgbs, path, fps=30):
    writer = iio.get_writer(path, fps=fps, format='FFMPEG', mode='I')
    for rgb in rgbs:
        writer.append_data(rgb)
    writer.close()