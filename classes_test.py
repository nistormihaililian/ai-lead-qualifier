class Lead:
    def __init__(self, name, email, score, source):
        self.name = name
        self.email = email
        self.score = score
        self.source = source
        self.status = "pending"

    def is_qualified(self):
        return self.score >= 70

    def summary(self):
        status = "Qualified" if self.is_qualified() else "Needs nurturing"
        return f"{self.name} ({self.source}) - {status}"

    def qualify(self):
        if self.is_qualified():
            self.status = "qualified"
        else:
            self.status = "nurture"


lead1 = Lead("Ion Popescu", "ion@firma.ro", 85, "Facebook Ads")

print(lead1.status)
lead1.qualify()
print(lead1.status)