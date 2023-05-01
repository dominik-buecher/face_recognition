def project_points(points, f):
    new_points = []

    new_points.append(f*(points[0]/points[2]))
    new_points.append(f*(points[1]/points[2]))


    return new_points




if __name__ == '__main__':

    p = [200, 200, 120]
    f = 50
    points = project_points(p, f)
    print("Image Koordiantes: ", points)




