Classes is a .csv file (duh)

Teachers is not, and is in competitive programming input format. This is because representing it as a csv is very long and painful, and it is probably faster to represent it as it is right now and convert it via computer prog if necessary

Format of teachers is:

n

Teacher
bit string representing periods can teach
# classes can teach followed by classes can teach (or just all)
# classes specifically excluded followed by excluded classes
# classes must teach followed by such classes
# sections needs to teach